render_script.py is the script of non-Lambertian object relighting. 

./coeffs includes the Spherical Harmonic coefficients of new lighting, it is pre-computed by fitting Spherical Harmonics to lighting panoramas. 

./cutmaps includes the cut-out backgrounds of the new environment for object insertion. 

./hdr includes the HDR lighting and its PNG visualization.

./obj_tree is the inverse rendering outputs of an example object image (the input image is included in the folder as well). 

For a quick demo, run render_script.py to generate a relit image. 

For references, there are also scripts for generating relighting videos (render_video.py) and naive insertion (naive_insertion.py) as in the supplementary video.  