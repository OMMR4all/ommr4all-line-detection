# Line Detection module of OMMR4all

Line segmentation algorithms for the OMMR4all project, originally created by Alexander Hartelt.


## Dependencies

This projects requires the pixel classifier of
https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/page-segmentation
to be installed locally.
 * Clone the page segmentation repository `git clone https://gitlab2.informatik.uni-wuerzburg.de/chw71yx/page-segmentation`
 * (Optional but recommended) Activate your virtual environment 
   (adapt to your path): `source venv/bin/activate`
 * install page segmentation `cd page-segmentation && python setup.py install`
 * the line detection is installed
 * install the specific requirements of line detection `cd ommr4all-line-detection && pip install -r requirements.txt`

 
 