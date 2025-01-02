# Traffic Prediction Model 🚗🚦

This project uses YOLOv8 and CNN (trained on various datasets) to predict car colors and count cars at traffic signals. The model also detects gender of people at the traffic signal, and counts other vehicles such as bikes and trucks.

## Features ✨

Predicts the color of cars at traffic signals (with red and blue cars swapped). 🔴🔵

Detects the number of cars in the traffic signal and displays them. 🚙

Identifies male and female pedestrians at the traffic signal. 🚶‍♂️🚶‍♀️

Counts other types of vehicles (e.g., trucks, bikes) at the traffic signal. 🚲🚚

## Technologies Used 🔧

YOLOv8 (PyTorch): For vehicle detection and segmentation.

CNN (Keras): For car color prediction and gender classification.

OpenCV: For video/image processing.

## Installation ⚙️
### Clone the repository:

git clone https://github.com/Hariarul/Smart-Traffic-Monitoring-System

### Install dependencies:

pip install -r requirements.txt

Download pre-trained models for YOLOv8 (vehicle detection), CNN (car color classification and gender detection).

### Run the application:

streamlit run traffic analyzer.py

## How It Works 🎬

Image/Video Upload: Upload images or videos of traffic signals, and the model predicts and displays:

Car count (number of cars detected).

Gender count (number of males and females detected).

Other vehicles count (such as trucks and bikes).

Detection and Swapping: The model identifies red and blue cars, swapping their colors accordingly (red cars turn blue and vice versa).

Vehicle Count: The model counts how many cars, other vehicles, and pedestrians (male/female) are present at the signal.

On-Screen Display: The results are shown directly on the screen, including the counts for cars, males, females, and other vehicles.

## Example Output 🧑‍🤝‍🧑

Input Video/Image: Traffic signal showing red and blue cars with pedestrians.

### Displayed Output on screen:

Cars: 5

Male Pedestrians: 5

Female Pedestrians: 3

Other Vehicles: 2

Input Video/Image: Traffic signal showing a mix of cars, trucks, and pedestrians.
### Displayed Output on screen:

Cars: 3

Male Pedestrians: 2

Female Pedestrians: 1

Other Vehicles: 3

## Results 📊

Accuracy: The model achieves good accuracy for car color detection, vehicle counting, and gender recognition under various traffic conditions.

Real-time Processing: The model can process 15-20 frames per second in real-time video streams.

On-Screen Display: Displays live counts and predictions for cars, pedestrians, and other vehicles on the screen.

Example Display:

Cars Detected: 5

Male Pedestrians: 5

Female Pedestrians: 3

Other Vehicles: 2
