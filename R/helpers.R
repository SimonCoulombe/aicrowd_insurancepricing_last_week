add_equal_weight_group <- function(table, sort_by, expo = NULL, group_variable_name = "groupe", nb = 10) {
  sort_by_var <- enquo(sort_by)
  groupe_variable_name_var <- enquo(group_variable_name)
  if (!(missing(expo))) { # https://stackoverflow.com/questions/48504942/testing-a-function-that-uses-enquo-for-a-null-parameter
    expo_var <- enquo(expo)
    total <- table %>%
      pull(!!expo_var) %>%
      sum()
    br <- seq(0, total, length.out = nb + 1) %>%
      head(-1) %>%
      c(Inf) %>%
      unique()
    table %>%
      arrange(!!sort_by_var) %>%
      mutate(cumExpo = cumsum(!!expo_var)) %>%
      mutate(!!group_variable_name := cut(cumExpo, breaks = br, ordered_result = TRUE, include.lowest = TRUE) %>% as.numeric()) %>%
      select(-cumExpo)
  } else {
    total <- nrow(table)
    br <- seq(0, total, length.out = nb + 1) %>%
      head(-1) %>%
      c(Inf) %>%
      unique()
    table %>%
      arrange(!!sort_by_var) %>%
      mutate(cumExpo = row_number()) %>%
      mutate(!!group_variable_name := cut(cumExpo, breaks = br, ordered_result = TRUE, include.lowest = TRUE) %>% as.numeric()) %>%
      select(-cumExpo)
  }
}

# define new functions ----
create__group_folds <- function(df, group_var = id_policy, n_folds = 5, seed = 42) {
  set.seed(seed)
  group_var_values <- df %>% 
    distinct({{ group_var }}) # everyone has 4 records, always put someone's  4 records in the same fold
  folds <- group_var_values[sample(nrow(group_var_values)), ] %>% # randomize policynumber order, then assign  folds according to row number
    mutate(fold = row_number() %% n_folds + 1)
}
