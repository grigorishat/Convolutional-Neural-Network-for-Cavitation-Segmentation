## Example images
Here are exemplary images from four different operating points. The images are stored in ``.tif`` format and can be read in and pre-processed with the script ``image_enhancer.py``, which is to be found in the folder 'development'. During image pre-processing, the insignificant areas above and below the hydrofoil are blackened. Therefore, the edges of the hydrofoils for all operating points are stored in ``edges.csv``. For post-processing after contour identification, it is important to rotate the images to ensure the flow direction is from left to right.

These operating points are included in the image folder: ($Re$ - Reynolds number, $\sigma$ - cavitation number)
- hssa1002: $Re = 8\times 10^5$, $\sigma=2.2$
- hssa1004: $Re = 8\times 10^5$, $\sigma = 2.6$
- hssa1006: $Re = 8\times 10^5$, $\sigma = 3.0 $
- hssa1008: $Re = 8\times 10^5$, $\sigma = 3.4 $
The images were recorded with a framerate of $18$ kFPS and an incidence of $12^\circ$.