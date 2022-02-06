In order to run the script the following steps need to be followed: 
- If it is not installed, install Jupyter Notebook or a similar SW in order to run .ipynb files. 
- Open with Jupyter Notebook the file named mask_to_coco_converter.ipynb. 
- Place the mask images that are wanted to transform to COCO under the folder dataset/annotations_mask/ 
- IMPORTANT: The images must be PNG files, otherwise the script would not work as expected. 
- Run each cell from above to below of the file mask_to_coco_converter.ipynb. 
- When the scripts finishes the execution, the annotations in COCO are located under the folder output/annotations.json. 
- If this tool is used complemented with the UI (v3), open the original images and the annotations.json to see the annotations in the images. 

Thank you for using the mask_to_coco_converter. 

AUTHORS: Arturo Pinar, Giuseppe Alessio Dimonte, Yueqiao Huang. 

ACKNOWLEDGEMENT & REFERENCES: 

This code is an adapted version of the code found in the Github repository
https://github.com/chrise96/image-to-coco-json-converter
