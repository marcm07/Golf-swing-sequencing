# Golf Swing Sequencing using Computer Vision

## Introduction
Analysis of golf swing events is a valuable tool to aid all golfers in improving their swing. Image processing and machine learning enable an automated system to perform golf swing sequencing using images. The majority of swing sequencing systems involve the use of expensive camera equipment or a motion capture suit. Therefore, an image-based swing classification system is proposed. It is evaluated on the GolfDB dataset. The system implements an automated golfer detector and traditional machine learning algorithms to classify swing events. 

This is Experiment 2 from my golf swing sequencing honours project focusing on classifying swing events using various classifcation models. 
Experiment 2 takes in the entire image without cropping the golfer from the image and subsequently learns the different golf swing events namely: 
  - Address
  - Toe-up
  - Mid-Backswing
  - Top
  - Mid-Downswing
  - Impact
  - Mid-Follow-Through
  - Finish
  
The images have already been already been read into numpy arrays with their target classes. 

The rest of the project resources and thesis can be found at the following link: https://sites.google.com/a/rucis.co.za/cs/research/g18m6731

## Running the Program
1. The input numpy files need to be downlaoded from the follwing google drive link: https://drive.google.com/file/d/1jRDOn1CNuY7JQOTcmbo_xnto5-UgZ0Bb/view?usp=sharing
2. These files need to be extracted to a file named "input"
3. Run the experiment_2_golfdb_entire_image_dataset.py file using python. 

System requirments: 
- Tensorflow 2.X

## Citation
Please cite the following if any of this work is used or helpful in your research:

Marais, M., Brown, D. (2022). Golf Swing Sequencing Using Computer Vision. In: Pinho, A.J., Georgieva, P., Teixeira, L.F., Sánchez, J.A. (eds) Pattern Recognition and Image Analysis. IbPRIA 2022. Lecture Notes in Computer Science, vol 13256. Springer, Cham. https://doi.org/10.1007/978-3-031-04881-4_28
