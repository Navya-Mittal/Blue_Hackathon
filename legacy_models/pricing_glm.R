# pricing_glm.R -- Risk-based loan pricing model (illustrative sample; input data not included)
# Linear model (Gaussian GLM) for the offered interest rate. Owner: Retail Pricing.

loans <- read.csv("loan_book.csv")

model <- glm(offered_rate ~ pd_score + loan_term + loan_amount_k + secured + region,
             data = loans, family = gaussian(link = "identity"))

loans$fitted_rate <- predict(model, newdata = loans, type = "response")
write.csv(loans, "priced_loans.csv", row.names = FALSE)
