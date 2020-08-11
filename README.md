# Face-Detection-and-Recognition
A unified embedding for face recognition and verification.
The model is based on the FaceNet Architecture.

Striving for an embedding f(x), from an image x into a feature space, such that the squared distance between all faces, independent of imaging conditions,
of same identity is small, whereas the distance between a pair of images from different identities is large.

#### Model Structure:-
<p align="center">
  <img src="Face%20Recognition/README_content/faceNet.png?raw=true" height="150" width=660" title="Model Architecture">
</p>

* Deep architecure model can be viewed in inception_block.py file.
* Alignment of the faces using HOG is obtained from the face_detection_landmarks_alignment.py file.

#### Face Recognition: Step By Step:-
1. Detect, transform and crop faces on input images.
2. Use the CNN to extract 128-dimensional representation, or embedding of faces from the aligned input images in embedding space, Euclidean distance directly corresponds to measure
of face similarity.
3. Compare input embedding vectors to labeled embedding vectors in a database. Here, a SVM and a KNN classifier, trained on labeled embedding vectors, play the role of a db.

#### Euclidean Distance is measured using Triplet loss:-
<p align="center">
  <img src="Face%20Recognition/README_content/tripletloss.png?raw=true" height="150" width=660" title="Triplet Loss">
  <img src="Face%20Recognition/README_content/triplet%20loss%20terms.png?raw=true" height="150" width=660" title="Triplet Loss">
 </p>
 
 #### Distances observed on different cirumstances:-
 <p align="center">
  <img src="Face%20Recognition/README_content/SamePeopleDistance.png?raw=true" height="250" width=660" title="Triplet Loss">
  <img src="Face%20Recognition/README_content/DifferentPeopleDistance.png?raw=true" height="250" width=660" title="Triplet Loss">
 </p>
 
*The KNN accuracy obtained is 94.915*

*The SVM accuracy obtained is 88.135*

#### T-distributed Stochastic Neighbor Embedding (t-SNE) visualization of the custom dataset after training.
<p align="center">
  <img src="Face%20Recognition/README_content/TSNE.png?raw=true" height="250" width=560" title="t-SNE">
</p>

#### Test of the Model in static images.
<p align="center">
  <img src="Face%20Recognition/README_content/FaceRecognitionInImage.jpg?raw=true" height="550" width=560" title="TestImage">
</p>

#### Test of the Model in webcam.
<p align="center">
  <img src="Face%20Recognition/README_content/webcam_check.png?raw=true" height="350" width=660" title="Test Image">
</p>

### Additional Component
**Build a custom face recognition dataset**

* build_face_dataset_webcam.py file can be used to build a custom face dataset using our webcam.
  1. Run the following command python build_face_dataset_webcam.py --cascade=haarcascade_frontalface_alt.xml --ouput=images/(name_of_the_person)
  2. After the webcam opens press key k to save the image in the folder specified above else press q to exit

* build_face_dataset_APIs.py can be used to build a custom face dataset using Microsoft API
  1. Run the following command python build_face_dataset_APIs.py --query=(image_search_query_to_use) --ouput=images/(name_of_the_person)
