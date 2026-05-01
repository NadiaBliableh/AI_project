# Smart Plant Watering Scheduler 🌿

An AI-based garden management system built entirely in pure Python 
(no external ML libraries) that predicts plant watering needs and 
optimizes the watering sequence.

## Features
- **Perceptron Classifier** — trained from scratch to predict whether 
  each plant needs water based on soil moisture, last watered time, 
  and plant type
- **Simulated Annealing Optimizer** — finds the most efficient watering 
  order minimizing total distance walked and missed plants
- **Interactive GUI** — place plants on a garden map, visualize 
  predictions, and watch the SA optimization step by step
- **Excel Export** — save full garden data with watering order to .xlsx

## Tech Stack
Python · Tkinter · Pure stdlib only (no numpy, pandas, or sklearn)

## How to Run
1. Clone the repo
2. Place `Data.xlsx` in the project folder
3. Run:
