# credit_pd.R -- Probability-of-Default (PD) model, retail credit card book
# Legacy production model (logistic regression). Owner: Retail Credit Risk.
# Usage: Rscript credit_pd.R <train.csv> <out_dir> <score1.csv> [<score2.csv> ...]

CUTOFF <- 0.30     # decline the application if PD >= CUTOFF

args        <- commandArgs(trailingOnly = TRUE)
train_path  <- args[1]
out_dir     <- args[2]
score_paths <- args[3:length(args)]

prep <- function(d) {
  # Business rules: collapse undocumented categories into "others"
  d$EDUCATION <- ifelse(d$EDUCATION %in% c(0, 5, 6), 4, d$EDUCATION)
  d$MARRIAGE  <- ifelse(d$MARRIAGE == 0, 3, d$MARRIAGE)
  d$EDUCATION <- factor(d$EDUCATION, levels = c(1, 2, 3, 4))
  d$MARRIAGE  <- factor(d$MARRIAGE,  levels = c(1, 2, 3))
  # Derived predictors
  d$log_limit  <- log(d$LIMIT_BAL)
  d$bill1_k    <- d$BILL_AMT1 / 1000
  d$payamt1_k  <- d$PAY_AMT1 / 1000
  d$payamt2_k  <- d$PAY_AMT2 / 1000
  d
}

train <- prep(read.csv(train_path))

model <- glm(default ~ log_limit + EDUCATION + MARRIAGE + PAY_1 + PAY_2 + PAY_3 +
               bill1_k + payamt1_k + payamt2_k,
             data = train, family = binomial(link = "logit"))

write.csv(data.frame(term = names(coef(model)), estimate = unname(coef(model))),
          file.path(out_dir, "coefficients.csv"), row.names = FALSE)

for (p in score_paths) {
  d   <- prep(read.csv(p))
  pd  <- predict(model, newdata = d, type = "response")
  out <- data.frame(id = d$id, pd = pd, decline = as.integer(pd >= CUTOFF))
  write.csv(out, file.path(out_dir, paste0("pred_", tools::file_path_sans_ext(basename(p)), ".csv")),
            row.names = FALSE)
}
