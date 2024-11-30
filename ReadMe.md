### Aviator Predictor
A Python-based real-time multiplier predictor for the 1Win Aviator game. It uses historical game data to train a machine learning model (LSTM) and provides live predictions through an interactive graphical user interface (GUI).

## Features
Fetches real-time game history and current state from the 1Win Aviator API.
Utilizes an LSTM model for predicting the next multiplier.
Dynamic GUI for displaying predictions and real-time updates.
Fully automated training and prediction process.
Requirements
Ensure the following are installed on your system:

Python 3.7 to 3.10 (TensorFlow requires this range)
64-bit Python (TensorFlow is incompatible with 32-bit Python)
Python Libraries
Install required libraries with:

bash
Copy code
pip install tensorflow pandas numpy requests
For minimal Python installations, you may also need:

bash
Copy code
sudo apt-get install python3-tk  # For Tkinter on Ubuntu/Debian
Installation
Clone this repository or download the script:

bash
Copy code
git clone https://github.com/TA-wiah/aviator-predictor.git
cd aviator-predictor
Install the dependencies:

bash
Copy code
pip install tensorflow pandas numpy requests
Run the program:

bash
Copy code
python aviator_predictor.py

## Usage
A splash screen appears with the title "Aviator Predictor by CyberJay".
The main GUI displays:
Current Multiplier: Fetched live from the game API.
Predicted Multiplier: Next predicted multiplier based on historical trends.
The program automatically fetches historical data, trains an LSTM model, and begins real-time predictions.

## Code Overview
fetch_history(): Retrieves historical game multipliers from the API.
fetch_current_state(): Fetches the current multiplier.
train_model(): Trains an LSTM model using historical data.
real_time_prediction(): Predicts the next multiplier and updates the GUI.
start_gui(): Launches the graphical interface.
Known Issues
TensorFlow Installation: Ensure you use a compatible Python version (3.7–3.10) and a 64-bit environment.
Game Randomness: Predictions are speculative due to game randomness.

### License
This project is open-source under the MIT License.

## Contributions
Feel free to fork the repository, suggest improvements, or raise issues via GitHub.

### Credits
Developed by CyberJay. 🎉