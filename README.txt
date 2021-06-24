shapes.dat = face detector, don't change

ear_counter.py = actual python code that analyzes video

blink_counter.xlsx = excel sheet that visualizes and counts blinks using output from ear_counter.py-- first tab counts using raw data, if there's baseline drift use second tab that detrends

Instructions: Put video (>5 min long, the clearer and more well-lit the face the better) into directory. Run ear_counter.py, which outputs a csv with a single row. Take that single row and put it into blink_counter.xlsx. You can figure the rest out. Use Record2.wmv (yours truly!) as a sample