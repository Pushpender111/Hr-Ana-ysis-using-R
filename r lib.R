# ============================================================

# IBM HR Attrition — Full R Analysis (Clean Version)

# ============================================================

library(tidyverse)
library(ggplot2)
library(scales)
library(corrplot)
library(caret)
library(randomForest)

# ── 1. LOAD DATA ─────────────────────────────────────────────

df <- read.csv("D:\Hr-Analytics\data\WA_Fn-UseC_-HR-Employee-Attrition.csv",
               stringsAsFactors = TRUE)

cat("Dimensions:", dim(df), "\n")
glimpse(df)
summary(df)

# ── 2. DATA CLEANING ──────────────────────────────────────────

colSums(is.na(df))

# Drop constant columns

df <- df %>% select(-EmployeeCount, -Over18, -StandardHours)

# Convert target to factor

df$Attrition <- factor(df$Attrition, levels = c("No", "Yes"))

# ── 3. ATTRITION OVERVIEW ─────────────────────────────────────

attrition_rate <- df %>%
  count(Attrition) %>%
  mutate(pct = n / sum(n) * 100)

print(attrition_rate)

ggplot(attrition_rate, aes(x = Attrition, y = pct, fill = Attrition)) +
  geom_col(width = 0.5) +
  geom_text(aes(label = sprintf("%.1f%%", pct)), vjust = -0.5, fontface = "bold") +
  scale_fill_manual(values = c("No" = "#2ecc71", "Yes" = "#e74c3c")) +
  labs(title = "Overall Attrition Rate", y = "Percentage (%)", x = NULL) +
  theme_minimal(base_size = 13) +
  theme(legend.position = "none")

# ── 4. ATTRITION BY CATEGORICAL VARIABLES ─────────────────────

cat_vars <- c("Department", "JobRole", "OverTime", "MaritalStatus",
              "BusinessTravel", "EducationField", "Gender")

for (var in cat_vars) {
  p <- df %>%
    group_by(.data[[var]], Attrition) %>%
    summarise(n = n(), .groups = "drop") %>%
    group_by(.data[[var]]) %>%
    mutate(pct = n / sum(n) * 100) %>%
    filter(Attrition == "Yes") %>%
    ggplot(aes(x = reorder(.data[[var]], pct), y = pct, fill = pct)) +
    geom_col() +
    coord_flip() +
    scale_fill_gradient(low = "#f9ca24", high = "#e74c3c") +
    labs(title = paste("Attrition Rate by", var), x = var, y = "Attrition %") +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none")
  print(p)
}

# ── 5. ATTRITION BY NUMERIC VARIABLES ─────────────────────────

num_vars <- c("Age", "MonthlyIncome", "YearsAtCompany",
              "TotalWorkingYears", "DistanceFromHome", "JobSatisfaction")

for (var in num_vars) {
  p <- ggplot(df, aes(x = Attrition, y = .data[[var]], fill = Attrition)) +
    geom_boxplot(alpha = 0.7, outlier.shape = 21) +
    scale_fill_manual(values = c("No" = "#3498db", "Yes" = "#e74c3c")) +
    labs(title = paste(var, "by Attrition"), y = var, x = NULL) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "none")
  print(p)
}

# ── 6. CORRELATION HEATMAP ────────────────────────────────────

num_df <- df %>% select(where(is.numeric))
cor_matrix <- cor(num_df, use = "complete.obs")

corrplot(cor_matrix,
         method  = "color",
         type    = "upper",
         tl.cex  = 0.7,
         tl.col  = "black",
         col     = colorRampPalette(c("#e74c3c", "white", "#2980b9"))(200),
         title   = "Numeric Feature Correlations",
         mar     = c(0, 0, 1, 0))

# ── 7. OVERTIME × DEPARTMENT INTERACTION ──────────────────────

df %>%
  group_by(Department, OverTime, Attrition) %>%
  summarise(n = n(), .groups = "drop") %>%
  group_by(Department, OverTime) %>%
  mutate(pct = n / sum(n) * 100) %>%
  filter(Attrition == "Yes") %>%
  ggplot(aes(x = Department, y = pct, fill = OverTime)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = c("No" = "#27ae60", "Yes" = "#c0392b")) +
  labs(title = "Attrition Rate: OverTime × Department",
       y = "Attrition %", fill = "OverTime") +
  theme_minimal(base_size = 12)

# ── 8. RANDOM FOREST — FEATURE IMPORTANCE ─────────────────────

set.seed(42)

train_idx <- createDataPartition(df$Attrition, p = 0.8, list = FALSE)
train_df  <- df[train_idx, ]
test_df   <- df[-train_idx, ]

rf_model <- randomForest(Attrition ~ ., data = train_df,
                         ntree = 300,
                         mtry = 5,
                         importance = TRUE)

preds <- predict(rf_model, test_df)
cm <- confusionMatrix(preds, test_df$Attrition, positive = "Yes")
print(cm)

importance_df <- as.data.frame(importance(rf_model)) %>%
  rownames_to_column("Feature") %>%
  arrange(desc(MeanDecreaseGini))

ggplot(importance_df[1:15, ],
       aes(x = reorder(Feature, MeanDecreaseGini), y = MeanDecreaseGini)) +
  geom_col(fill = "#8e44ad") +
  coord_flip() +
  labs(title = "Top 15 Features — Random Forest Importance",
       x = NULL, y = "Mean Decrease Gini") +
  theme_minimal(base_size = 12)

# ── 9. SUMMARY TABLE ──────────────────────────────────────────

df %>%
  group_by(Department, JobRole) %>%
  summarise(
    n = n(),
    attrition = mean(Attrition == "Yes") * 100,
    avg_income = mean(MonthlyIncome),
    avg_age = mean(Age),
    .groups = "drop"
  ) %>%
  arrange(desc(attrition)) %>%
  print(n = 20)

