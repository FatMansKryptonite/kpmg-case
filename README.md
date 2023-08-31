# TODOs
- [x] Clean dates
- [x] Join missing_report
- [x] Make profiles
- [x] Make pairplot
- [x] Join shift_report
- [x] Group location data and make use of ZIP code
- [x] Handle categorical in random forest
- [x] There are no doctors in the test set.....
- [x] __Balance dataset!!__
- [ ] Generate synthetic data for missing values
  - Make it group specific (for positive/negative samples)

# Sporadic Thoughts and Issues
- Dataset highly imbalanced
  - Dropping nan-values removes half of the positive samples -> more imbalance
- It is evident from the pairplot that short, heavy people are abducted to a greater extent 
(could a sadistic doctor be our culprit?!)
- Location is slightly correlated with missing status, let's group! 
  - Why would they provide "zip codes" for these countries? Surely there is no relationship between the numerical value of 
  the zip code and the probability of going missing?


# Results
- Anthony Vonnegut is abducting short, heavy people for his sick little experiments...

# Further work
