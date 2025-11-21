# Assignment 6 Part 1 - Writeup

**Name:** _______________  
**Date:** _______________

---

## Part 1: Understanding Your Model

### Question 1: R² Score Interpretation
What does the R² score tell you about your model? What does it mean if R² is close to 1? What if it's close to 0?

I tells me that there is a high correlation between studying hours and test scores. Since it is close to one, the model is accurate but if it was close to 0, than it would be a bad model and have bad correlation. 




---

### Question 2: Mean Squared Error (MSE)
What does the MSE (Mean Squared Error) mean in plain English? Why do you think we square the errors instead of just taking the average of the errors?

The Mean Squared Error tells me, on average, how far off my predictions are from the real scores. The bigger the number, the worse the predictions. We square the errors so that big mistakes count more than small ones, and so negative and positive errors don’t cancel each other out. If we didn’t square them, a prediction that’s too high and one that’s too low could average to zero even though the model made mistakes.




---

### Question 3: Model Reliability
Would you trust this model to predict a score for a student who studied 10 hours? Why or why not? Consider:
- What's the maximum hours in your dataset?
- What happens when you make predictions outside the range of your training data?

I would be a little careful about predicting for a student who studied 10 hours. The highest number of hours in the dataset is around 9.6, which is close, but still outside the range of the training data.
When you predict outside the range of your data, the model might guess something unrealistic because it has never seen that kind of example before.




---

## Part 2: Data Analysis

### Question 4: Relationship Description
Looking at your scatter plot, describe the relationship between hours studied and test scores. Is it:
- Strong or weak?
- Linear or non-linear?
- Positive or negative?

The scatter plot shows a strong, positive, and mostly linear relationship. As the number of hours studied increases, the test score also increases in a predictable way. The points are fairly close to a line, which means the pattern is strong.




---

### Question 5: Real-World Limitations
What are some real-world factors that could affect test scores that this model doesn't account for? List at least 3 factors.

**YOUR ANSWER:**
1. How difficult the test was
2. How well the student slept or their stress level
3. Intelligence or prior knowledge in the subject


---

## Part 3: Code Reflection

### Question 6: Train/Test Split
Why do we split our data into training and testing sets? What would happen if we trained and tested on the same data?

We split the data so the model can learn from one part and be tested on data it has never seen before. If we trained and tested on the same data, the model would look perfect, but only because it already saw the answers.




---

### Question 7: Most Challenging Part
What was the most challenging part of this assignment for you? How did you overcome it (or what help do you still need)?

The most challenging part of the assignment was the beginning because I didn't really understand what to type in to finish te function. After finishing one though, I started to understand what to do. 




---

## Part 4: Extending Your Learning

### Question 8: Future Applications
Describe one real-world problem you could solve with linear regression. What would be your:
- **Feature (X):** 
- **Target (Y):** 
- **Why this relationship might be linear:**

A real-world problem I could solve with linear regression is predicting how many minutes it will take to get to school based on distance.

Feature (X): distance from home to school

Target (Y): travel time

Why it might be linear:
Usually, the farther you have to travel, the longer it takes. It’s not perfect, but it’s a relationship that is mostly straight-line for everyday travel.




---

## Grading Checklist (for your reference)

Before submitting, make sure you have:
- [ ] Completed all functions in `a6_part1.py`
- [ ] Generated and saved `scatter_plot.png`
- [ ] Generated and saved `predictions_plot.png`
- [ ] Answered all questions in this writeup with thoughtful responses
- [ ] Pushed all files to GitHub (code, plots, and this writeup)

---

## Optional: Extra Credit (+2 points)

If you want to challenge yourself, modify your code to:
1. Try different train/test split ratios (60/40, 70/30, 90/10)
2. Record the R² score for each split
3. Explain below which split ratio worked best and why you think that is

**YOUR ANSWER:**
