# Cat Expression Matcher

A real-time facial expression detection system that matches your expressions with cat images!

## Features

- Real-time face detection using your laptop camera
- Expression recognition for:
  - **Tongue Out** → Shows `toung out.jpeg`
  - **Shocked** → Shows `shocked.jpeg`
  - **Staring** → Shows `staring.jpeg`
  - **Side Look** → Shows `giving side look.jpeg`
- Displays matching cat image in real-time
- Uses MediaPipe Face Mesh for accurate facial landmark detection

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the script:
```bash
python cat_expression_matcher.py
```

2. Make facial expressions in front of your camera:
   - **Stick your tongue out** to see the tongue-out cat
   - **Open your eyes wide and mouth** for the shocked cat
   - **Turn your head to the side** for the side-look cat
   - **Look straight with focused eyes** for the staring cat

3. The matching cat image will appear in the top-right corner of the camera feed

4. Press `q` to quit

## How It Works

The script uses MediaPipe Face Mesh to detect 468 facial landmarks. It analyzes:
- **Mouth openness**: Detects when mouth is wide open (tongue out)
- **Eye openness**: Detects wide eyes (shocked expression)
- **Head rotation**: Detects when head is turned to the side
- **Default**: Staring expression when other conditions aren't met

The expression must be detected consistently for 5 frames before switching to ensure stability.
