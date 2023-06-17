train_spec_separation.py is the self-supervised training scripts of specular branch. 

test_spec_separation.py is the corresponding testing script, which should be ready to run with the provided testing data. 

train_end2end_invrender.py is the self-supervised training scripts of Normal-Net and Light-Net (diffuse branch). 

test_end2end_invrender.py is the corresponding testing script, which should be ready to run with the provided testing data. 

For diffuse object, run the diffuse branch directly. For glossy objects, run Spec-Net first. 

The training dataset is the Relit dataset we captured and collected, which is a large-scale foreground-aligned video dataset. It is released on the project page. 