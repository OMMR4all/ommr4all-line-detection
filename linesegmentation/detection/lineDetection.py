### misc imports
import sys, os, cv2, operator, tqdm, multiprocessing
from functools import partial
from typing import List, Generator, Optional
from itertools import tee
from dataclasses import dataclass
import numpy as np
from matplotlib import pyplot as plt

### image specific imports
from PIL import Image
from scipy import misc
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from scipy.interpolate import interp1d, interpolate

### project specific imports
from pagesegmentation.lib.predictor import PredictSettings
from linesegmentation.preprocessing.binarization.ocropus_binarizer import binarize
from linesegmentation.preprocessing.enhancing.enhancer import enhance
from linesegmentation.pixelclassifier.predictor import PCPredictor

@dataclass
class ImageData:
        path: str = None
        height: int = None
        image: np.array = None
        horizontalRunsImg: np.array = None
        staffLineHeight: int = None
        staffSpaceHeight: int = None

@dataclass
class LineDetectionSettings:
    numLine: int = 4
    minLength: int = 6
    lineExtension: bool = True
    debug: bool = False
    lineSpaceHeight: int = None
    targetLineSpaceHeight: int = None
    model: Optional[str] = None
    processes: int = 12

def createData(path, lineSpaceHeight):
    spaceHeight = lineSpaceHeight
    if lineSpaceHeight == 0:
        spaceHeight = verticalRuns(binarize(np.array(Image.open(path)) / 255))[0]
    imagedata = ImageData(path = path, height = spaceHeight)
    return imagedata

def verticalRuns(img: np.array):
    img = np.transpose(img)
    h = img.shape[0]
    w = img.shape[1]
    transitions = np.transpose(np.nonzero(np.diff(img)))
    white_runs = [0] * (w + 1)
    black_runs = [0] * (w + 1)
    a,b = tee(transitions)
    next(b,[])
    for f,g in zip(a,b):
        if f[0] != g[0]:
            continue
        tlen = g[1] - f[1]
        if img[f[0],f[1] + 1] == 1:
            white_runs[tlen] += 1
        else:
            black_runs[tlen] += 1

    for y in range(h):
        x = 1
        col = img[y,0]
        while x < w and img[y,x] == col:
            x+=1
        if col == 1:
            white_runs[x] += 1
        else:
            black_runs[x] += 1

        x = w  - 2
        col = img[y,w-1]
        while x >= 0 and img[y,x] == col:
            x -= 1
        if col == 1:
            white_runs[w - 1 - x] += 1
        else:
            black_runs[w - 1 - x] += 1
    black_r = np.argmax(black_runs) + 1
    # on pages with a lot of text the staffspaceheigth can be falsified.
    # --> skip the first elements of the array
    white_r = np.argmax(white_runs[black_r:]) + 1 + black_r
    img = np.transpose(img)
    return white_r,black_r

def calculateHorizontalRuns(img: np.array, minLength: int):
        h = img.shape[0]
        w = img.shape[1]
        npMatrix = np.zeros([h,w], dtype = np.uint8)
        t = np.transpose(np.nonzero(np.diff(img) == -1))
        for trans in t:
            y,x = trans[0],trans[1] + 1
            xo = x
            rl = 0
            while x < w and img[y,x] == 0:
                x += 1
            rl = x - xo
            if rl >= minLength:
                for x in range(xo,xo + rl):
                    npMatrix[y,x] = 255
        return npMatrix

class LineDetection:
    """Line detection class

    Attributes
    ----------
    settings : LineDetectionSettings
        Setting for the line detection algorithm
    predictor : PCPredictor, optional
        Necessary if the NN should be used for the binarisation

    """
    def __init__(self, settings: LineDetectionSettings):
        """Constructor of the LineDetection class

        Parameters
        ----------
        settings: LineDetectionSettings
            Settings for the line detection algorithm
        """
        self.settings = settings
        self.predictor = None
        if settings.model:
            pcsettings = PredictSettings(
                mode='meta',
                network=os.path.abspath(settings.model),
                output=None,
                high_res_output=False
            )
            self.predictor = PCPredictor(pcsettings, settings.targetLineSpaceHeight)

    def detect(self, imagePaths: List[str]) -> Generator[List[List[List[int]]], None, None]:
        """
        Function  to detect die stafflines in an image

        Parameters
        ----------
        settings: List[str]
            Paths to the images, which should be processed

        Yields
        ------
        List     [List    [List      [int]]]
        System   Staff    Polyline    y,x
        
            Example
            --------
            ####### Structure ######
            pointList[
                       system1[
                              staff1[
                                   [y1, x1]
                                   [y2, x2]
                                   ]
                              staff2[
                                     ...
                                   ]
                       system2[
                               ...
                             ]
                     ]    
        """
        if not self.settings.model:
            return self.detectbasic(imagePaths)
        else:
            return self.detectNN(imagePaths)

    def detectbasic(self, imagePaths: List[str]) -> Generator[List[List[List[int]]], None, None]:
        for imgPath in imagePaths:
            imageData = ImageData(path = imgPath)
            imageData.image = np.array(Image.open(imgPath)) / 255
            gray = imageData.image
            if (np.sum(np.histogram(gray)[0][1:-2]) != 0):
                gray = enhance(imageData.image)
            binary = binarize(gray)
            binarized = 1 - binary
            morph = binary_erosion(binarized, structure=np.full((5, 1), 1))
            morph = binary_dilation(morph, structure=np.full((5, 1), 1))
            staffs = (binarized ^ morph)
            imageData.staffSpaceHeight, imageData.staffLineHeight = verticalRuns(binary)
            imageData.horizontalRunsImg = calculateHorizontalRuns((1 - staffs), self.settings.minLength)
            yield self.__detectStaffLines(imageData)



    def detectNN(self, imagePaths: List[str]) -> Generator[List[List[List[int]]], None, None]:

        # print('Preparing image data')
        createDatap = partial(createData, lineSpaceHeight = self.settings.lineSpaceHeight)
        with multiprocessing.Pool(processes = self.settings.processes) as p:
            data = [v for v in tqdm.tqdm(p.imap(createDatap, imagePaths), total=len(imagePaths))]

        for i, pred in enumerate(self.predictor.predict(data)):
            data[i].staffSpaceHeight, data[i].staffLineHeight = verticalRuns(1 - pred)
            data[i].horizontalRunsImg = calculateHorizontalRuns((1 - (pred / 255)), self.settings.minLength)
            yield self.__detectStaffLines(data[i])

    def __detectStaffLines(self, imageData: ImageData):
        img = imageData.horizontalRunsImg
        stafflineHeight = imageData.staffLineHeight
        staffSpaceHeight = imageData.staffSpaceHeight

        connectivity = 8
        output = cv2.connectedComponentsWithStats(img, connectivity)

        ccList = []
        ## Calculate the position of all CCs
        for i in range(1,output[0]):
            cc = np.argwhere(output[1]==i)
            sortedCC = cc[cc[:,1].argsort()]
            ccList.append(sortedCC.tolist())

        ## Normalize the CCs (line segments), so that the height of each cc is normalized to one pixel
        def normalize(point_list):
            def insertIntoDict(key,value,aDict):
                if not key in aDict:
                    aDict[key] = [value]
                else:
                    aDict[key].append(value)
            n_point_list = []
            for cc in point_list:
                staff = {}
                for y, x in cc:
                    insertIntoDict(x, y, staff)
                staffs = []
                for key, value in staff.items():
                    staffs.append([int(np.floor(np.mean(value))), key])
                n_point_list.append(staffs)
            return n_point_list
        ccList = normalize(ccList)

        def connectCC(ccList, inplace = True):
            def pruneCC(cc, length):
                ccList = []
                for i in cc:
                    if abs(i[0][1] - i[-1][1]) > length:
                        ccList.append(i)
                return ccList
            def connect(max_dists: List[int], vert_dist: int, ccListcp):
                for max_dist in max_dists:
                    i = 0
                    while i < len(ccListcp):
                        l1 = ccListcp[i]
                        y1b, x1b = l1[0]
                        y1e, x1e = l1[-1]

                        found = False
                        for i2 in range(i + 1,len(ccListcp)):
                            l2 = ccListcp[i2]
                            y2b, x2b = l2[0]
                            y2e, x2e = l2[-1]
                            if x1e < x2b and x2b - x1e < max_dist:
                                distance = x2b - x1e
                                if np.abs(y1e - y2b) < vert_dist:
                                    ccListcp[i] = l1 + l2
                                    del ccListcp[i2]
                                    found = True
                                    break
                            elif x2e < x1b and x1b - x2e < max_dist:
                                if np.abs(y1b - y2e) < vert_dist:
                                    ccListcp[i] = l2 + l1
                                    del ccListcp[i2]
                                    found = True
                                    break
                        if not found:
                            i += 1
                    if (vert_dist == 2 and max_dist == 30):
                        ccListcp = pruneCC(ccListcp, 10)
                return ccListcp
            ccListcp = ccList

            if inplace != True:
                ccListcp = ccList.copy()

            for x in [[10, 30, 50, 100], [200, 300, 500]]:
                for vert_dist in [2, stafflineHeight, staffSpaceHeight / 5 + stafflineHeight, staffSpaceHeight / 3 + stafflineHeight]:
                    ccListcp = connect(x, vert_dist, ccListcp)
            return ccListcp

        line_List = connectCC(ccList)
        ## Remove lines which are shorter than 50px
        line_List = [l for l in line_List if l[-1][1] - l[0][1] > 50]
        ## Calculate medium height of all staffs
        mediumStaffHeight = [np.mean([y for y, x in staff]) for staff in line_List]


        ############ Debug #############
        staff2 = line_List.copy()
        ################################

        def pruneSmallLines(line_List, mediumStaffHeight, inplace = True):
            line_Listcp = line_List
            mediumStaffHeightcp = mediumStaffHeight
            if inplace != True:
                mediumStaffHeightcp = mediumStaffHeight.copy()
                line_Listcp = line_List.copy()
            while True:
                prevStaffh = 0
                for staff_ind, staffh in enumerate(mediumStaffHeightcp):
                    if (abs(prevStaffh - staffh) < staffSpaceHeight / 3.0) and prevStaffh != 0:
                        y1a, x1a = line_Listcp[staff_ind - 1][0]
                        y1e, x1e = line_Listcp[staff_ind - 1][-1]
                        y2a, x2a = line_Listcp[staff_ind][0]
                        y2e, x2e = line_Listcp[staff_ind][-1]
                        if (x2e >= x1e and x2a <= x1a):
                            del line_Listcp[staff_ind - 1]
                            del mediumStaffHeight[staff_ind - 1]
                            break
                        if (x2e <= x1e and x2a >= x1a):
                            del line_Listcp[staff_ind]
                            del mediumStaffHeightcp[staff_ind]
                            break
                        if (x2e >= x1e and x2a >= x1e ):
                            line_Listcp[staff_ind - 1] =  line_Listcp[staff_ind - 1] + line_Listcp[staff_ind]
                            del line_Listcp[staff_ind]
                            del mediumStaffHeightcp[staff_ind]
                            break
                        if (x2e <= x1e and x1a >= x2e):
                            line_Listcp[staff_ind - 1] =   line_Listcp[staff_ind] + line_Listcp[staff_ind - 1]
                            del line_Listcp[staff_ind]
                            del mediumStaffHeightcp[staff_ind]
                            break
                    prevStaffh = staffh
                    x = False
                else: break
            return line_Listcp, mediumStaffHeightcp

        line_List, mediumStaffHeight = pruneSmallLines(line_List, mediumStaffHeight, inplace = True)

        if self.settings.numLine != 0:
            staffindices = []
            for i, medium_y in enumerate(mediumStaffHeight):
                system = []
                if i in sum(staffindices, []):
                    continue
                height = medium_y
                for z, center_ys in enumerate(mediumStaffHeight):
                    if np.abs(height - center_ys) < 1.3 *(staffSpaceHeight + stafflineHeight):
                        system.append(z)
                        height = center_ys
                staffindices.append(system)
            staffindices = [staff for staff in staffindices if len(staff) >= 3]

            def blacknessOfLine(line, img):
                y, x = zip(*line)
                f = interpolate.interp1d(x,y)
                xStart, xEnd = x[0], x[-1]
                spacedNumbers = np.linspace(xStart, xEnd, num=int(abs(x[0] - x[-1]) * 1 / 5), endpoint=True)
                blackness = 0
                for i in spacedNumbers:
                    if img[int(f(i))][int(i)] == 255:
                        blackness += 1
                return blackness

            ## Remove the lines with the lowest blackness value in each system, so that len(staffs) <= numLine
            prune = True
            while prune == True:
                prune = False
                for staff_ind, staff in enumerate(staffindices):
                    if len(staff) > self.settings.numLine:
                        intensityOfStaff = {}
                        for line_ind, line in enumerate(staff):
                            intensityOfStaff[line_ind] = blacknessOfLine(line_List[line], img)
                        if intensityOfStaff:
                            prune = True
                            minBlackness = min(intensityOfStaff.items(), key = lambda t: t[1])
                            if minBlackness[0] == 0 or minBlackness[0] == len(intensityOfStaff):
                                del staffindices[staff_ind][minBlackness[0]]
                                del intensityOfStaff[minBlackness[0]]
                                continue
                            if len(staff) >= self.settings.numLine * 2 + 1 and self.settings.numLine != 0:
                                if len(staff[:minBlackness[0]]) > 2:
                                    staffindices.append(staff[:minBlackness[0]])
                                if len(staff[minBlackness[0]:]) > 2:
                                    staffindices.append(staff[minBlackness[0]:])
                                del staffindices[staff_ind]
                                continue
                            del staffindices[staff_ind][minBlackness[0]]
                            del intensityOfStaff[minBlackness[0]]

            staffList = []
            for z in staffindices:
                system = []
                for x in z:
                    system.append(line_List[x])
                staffList.append(system)

            if self.settings.lineExtension == True:

                for z_ind, z in enumerate(staffList):
                    sxs = [line[0][1] for line in z]
                    exs = [line[-1][1] for line in z]
                    minIndex_sxs, sxb = sxs.index(min(sxs)), min(sxs)
                    maxIndex_exs, exb = exs.index(max(exs)), max(exs)
                    ymi, xmi = zip(*z[minIndex_sxs])
                    minf = interpolate.interp1d(xmi,ymi, fill_value='extrapolate')
                    yma, xma = zip(*z[maxIndex_exs])
                    maxf = interpolate.interp1d(xma,yma, fill_value='extrapolate')

                    for line_ind, line in enumerate(z):
                        y, x = zip(*line)
                        if line[0][1] > xmi[0] and abs(line[0][1] - xmi[0]) > 5:
                            xStart, xEnd = xmi[0], min(line[0][1], z[minIndex_sxs][-1][1])
                            spacedNumbers = np.linspace(xStart, xEnd - 1, num=abs(xEnd - xStart) * 1/5, endpoint=True)
                            staffextension = []
                            if line[0][1] > xmi[-1]:
                                dif = minf(xma[-1]) - line[0][0]
                            else:
                                dif = minf(line[0][1]) - line[0][0]
                            for i in spacedNumbers:
                                staffextension.append([int(minf(i) - dif), int(i)])
                            if staffextension:
                                staffList[z_ind][line_ind] = staffextension + staffList[z_ind][line_ind]
                        if line[-1][1] < exs[maxIndex_exs] and abs(line[-1][1] - exs[maxIndex_exs]) > 5:
                            xStart, xEnd = max(line[-1][1], z[maxIndex_exs][0][1]), exs[maxIndex_exs]
                            spacedNumbers = np.linspace(xStart, xEnd , num=abs(xEnd - xStart) * 1/5, endpoint=True)
                            staffextension = []
                            if line[-1][1] < xma[0]:
                                dif = maxf(xma[0]) - line[-1][0]
                            else:
                                dif = maxf(line[-1][1]) - line[-1][0]
                            for i in spacedNumbers:
                                staffextension.append([int(maxf(i) - dif), int(i)])
                            if staffextension:
                                staffList[z_ind][line_ind] = staffList[z_ind][line_ind] + staffextension
                        if line[0][1] < sxb and abs(line[0][1] - sxs[minIndex_sxs]) > 5:
                            while len(line) > 0 and line[0][1] <= sxb:
                                del line[0]
                        if x[-1] > exb and abs(x[-1] - sxs[minIndex_sxs]) > 5:
                            while line[-1][1] >= exb:
                                del line[-1]

                for staff_ind, staffs in enumerate(staffList):
                    mediumStaffHeighOfLine = [np.mean([y for y, x in line]) for line in staffs]
                    while True:
                        prevLineh = 0
                        for line_ind, lineh in enumerate(mediumStaffHeighOfLine):
                            if (abs(prevLineh - lineh) < staffSpaceHeight / 2.0) and prevLineh != 0:
                                blackness1 = blacknessOfLine(staffList[staff_ind][line_ind], img)
                                blackness2 = blacknessOfLine(staffList[staff_ind][line_ind - 1], img)
                                if blackness1 > blackness2:
                                    del staffList[staff_ind][line_ind - 1]
                                    del mediumStaffHeighOfLine[line_ind -1]
                                    break
                                else:
                                    del staffList[staff_ind][line_ind]
                                    del mediumStaffHeighOfLine[line_ind]
                                    break
                            prevLineh = lineh
                            x = False
                        else: break

    ################# Debug ###################
        if (self.settings.debug):
            im = plt.imread(imageData.path)
            f, ax = plt.subplots(1, 3, True, True)
            ax[0].imshow(im, cmap = 'gray')
            cmap = plt.get_cmap('jet')
            colors = cmap(np.linspace(0, 1.0, len(staffList)))
            for system, color in zip(staffList,colors):
                for staff in system:
                    y, x = zip(*staff)
                    ax[0].plot(x,y, color = color)
            ax[1].imshow(img, cmap = 'gray')
            ax[2].imshow(im, cmap = 'gray')
            for staff in staff2:
                y, x = zip(*staff)
                ax[2].plot(x,y, 'r')
            plt.show()
    ############################################
        return staffList
