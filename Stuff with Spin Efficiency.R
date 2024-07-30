#Package Loading---------------------------------------------------------------
devtools::install_github("BillPetti/baseballr")
library(dplyr)
library(berryFunctions)
library(Boruta)
library(randomForest)
library(Metrics)
library(factoextra)
library(ggplot2)
library(caret)
library(xgboost)
library(pROC)
library(baseballr)
library(writexl)

#Train Data Loading------------------------------------------------------------
#Loading 2023 Data by Week
week1 <- statcast_search_pitchers(start_date = '2023-03-30', 
                                  end_date = '2023-04-06',
                                  pitcherid = NULL)
week2 <- statcast_search_pitchers(start_date = '2023-04-07', 
                                  end_date = '2023-04-14',
                                  pitcherid = NULL)
week3 <- statcast_search_pitchers(start_date = '2023-04-15', 
                                  end_date = '2023-04-22',
                                  pitcherid = NULL)
week4 <- statcast_search_pitchers(start_date = '2023-04-23', 
                                  end_date = '2023-04-30',
                                  pitcherid = NULL)
week5 <- statcast_search_pitchers(start_date = '2023-05-01', 
                                  end_date = '2023-05-08',
                                  pitcherid = NULL)
week6 <- statcast_search_pitchers(start_date = '2023-05-09', 
                                  end_date = '2023-05-16',
                                  pitcherid = NULL)
week7 <- statcast_search_pitchers(start_date = '2023-05-17', 
                                  end_date = '2023-05-24',
                                  pitcherid = NULL)
week8 <- statcast_search_pitchers(start_date = '2023-05-25', 
                                  end_date = '2023-06-01',
                                  pitcherid = NULL)
week9 <- statcast_search_pitchers(start_date = '2023-06-02', 
                                  end_date = '2023-06-09',
                                  pitcherid = NULL)
week10 <- statcast_search_pitchers(start_date = '2023-06-10', 
                                   end_date = '2023-06-17',
                                   pitcherid = NULL)
week11 <- statcast_search_pitchers(start_date = '2023-06-18', 
                                   end_date = '2023-06-25',
                                   pitcherid = NULL)
week12 <- statcast_search_pitchers(start_date = '2023-06-26', 
                                   end_date = '2023-07-03',
                                   pitcherid = NULL)
week13 <- statcast_search_pitchers(start_date = '2023-07-04', 
                                   end_date = '2023-07-11',
                                   pitcherid = NULL)
week14 <- statcast_search_pitchers(start_date = '2023-07-12', 
                                   end_date = '2023-07-19',
                                   pitcherid = NULL)
week15 <- statcast_search_pitchers(start_date = '2023-07-20', 
                                   end_date = '2023-07-27',
                                   pitcherid = NULL)
week16 <- statcast_search_pitchers(start_date = '2023-07-28', 
                                   end_date = '2023-08-04',
                                   pitcherid = NULL)
week17 <- statcast_search_pitchers(start_date = '2023-08-05', 
                                   end_date = '2023-08-12',
                                   pitcherid = NULL)
week18 <- statcast_search_pitchers(start_date = '2023-08-13', 
                                   end_date = '2023-08-20',
                                   pitcherid = NULL)
week19 <- statcast_search_pitchers(start_date = '2023-08-21', 
                                   end_date = '2023-08-28',
                                   pitcherid = NULL)
week20 <- statcast_search_pitchers(start_date = '2023-08-29', 
                                   end_date = '2023-09-05',
                                   pitcherid = NULL)
week21 <- statcast_search_pitchers(start_date = '2023-09-06', 
                                   end_date = '2023-09-13',
                                   pitcherid = NULL)
week22 <- statcast_search_pitchers(start_date = '2023-09-14', 
                                   end_date = '2023-09-21',
                                   pitcherid = NULL)
week23 <- statcast_search_pitchers(start_date = '2023-09-22', 
                                   end_date = '2023-09-29',
                                   pitcherid = NULL)
week24 <- statcast_search_pitchers(start_date = '2023-09-30', 
                                   end_date = '2023-10-07',
                                   pitcherid = NULL)

#Combining Weeks into Master Season Data
train_full <- rbind(week1, week2, week3, week4, week5, week6, week7, week8,
                       week9, week10, week11, week12, week13, week14, week15, 
                       week16, week17, week18, week19, week20, week21, week22,
                       week23, week24)
rm(week1, week2, week3, week4, week5, week6, week7, week8, week9, week10, 
   week11, week12, week13, week14, week15, week16, week17, week18, week19, 
   week20, week21, week22, week23, week24)

#Calculating Estimated Spin Efficiency Per Pitch
train_full$spin_axis_rad <- train_full$spin_axis * pi / 180

train_full <- train_full %>%
  mutate(TSM = release_spin_rate * sin(spin_axis_rad),
         G = release_spin_rate * cos(spin_axis_rad))

train_full <- train_full %>%
  mutate(spin_efficiency = (TSM / release_spin_rate) * 100)
min_spin_efficiency <- min(train_full$spin_efficiency, na.rm = TRUE)
max_spin_efficiency <- max(train_full$spin_efficiency, na.rm = TRUE)

train_full <- train_full %>%
  mutate(normalized_spin_efficiency = 
    100*(spin_efficiency - min_spin_efficiency) / (max_spin_efficiency - min_spin_efficiency)) 
rm(max_spin_efficiency, min_spin_efficiency)

#Calculating Vertical Approach Angle Per Pitch
train_full$vy_f <- -sqrt(train_full$vy0^2 - (2*train_full$ay*(50-(17/12))))
train_full$t <- (train_full$vy_f - train_full$vy0)/train_full$ay
train_full$vz_f <- train_full$vz0 + (train_full$az*train_full$t)
train_full$VAA <- -atan(train_full$vz_f/train_full$vy_f)*(180/pi)

#Paring Down Master Data to Key Specs
train_specs <- subset(train_full, select = c('player_name', 'p_throws',
                                             'description', 'pitch_type', 
                                             'release_speed', 'pfx_z', 'pfx_x', 
                                             'release_pos_z','release_extension', 
                                             'release_spin_rate', 
                                             'normalized_spin_efficiency',
                                             'VAA'))

#Test Data Loading-------------------------------------------------------------
#Loading Daily Test Data
test_full <- statcast_search_pitchers(start_date = Sys.Date()-1,
                                      end_date = Sys.Date(),
                                      pitcherid = NULL)

#Loading Full Season Test Data
week1 <- statcast_search_pitchers(start_date = '2024-03-30', 
                                  end_date = '2024-04-06',
                                  pitcherid = NULL)
week2 <- statcast_search_pitchers(start_date = '2024-04-07',
                                  end_date = '2024-04-14',
                                  pitcherid = NULL)
week3 <- statcast_search_pitchers(start_date = '2024-04-15', 
                                  end_date = '2024-04-22',
                                  pitcherid = NULL)
week4 <- statcast_search_pitchers(start_date = '2024-04-23', 
                                  end_date = '2024-04-30',
                                  pitcherid = NULL)
week5 <- statcast_search_pitchers(start_date = '2024-05-01', 
                                  end_date = '2024-05-08',
                                  pitcherid = NULL)
week6 <- statcast_search_pitchers(start_date = '2024-05-09', 
                                  end_date = '2024-05-16',
                                  pitcherid = NULL)
week7 <- statcast_search_pitchers(start_date = '2024-05-17', 
                                  end_date = '2024-05-24',
                                  pitcherid = NULL)
week8 <- statcast_search_pitchers(start_date = '2024-05-25', 
                                  end_date = '2024-06-01',
                                  pitcherid = NULL)
week9 <- statcast_search_pitchers(start_date = '2024-06-02',
                                  end_date = '2024-06-09',
                                  pitcherid = NULL)

fullseason_test <- rbind(week1, week2, week3, week4, week5, week6, week7, week8)
rm(week1, week2, week3, week4, week5, week6, week7, week8, week9)

#Calculating Normalized Spin Efficiency
test_full$spin_axis_rad <- test_full$spin_axis * pi / 180
test_full <- test_full %>%
  mutate(TSM = release_spin_rate * sin(spin_axis_rad),
         G = release_spin_rate * cos(spin_axis_rad))
test_full <- test_full %>%
  mutate(spin_efficiency = (TSM / release_spin_rate) * 100)
min_spin_efficiency <- min(test_full$spin_efficiency, na.rm = TRUE)
max_spin_efficiency <- max(test_full$spin_efficiency, na.rm = TRUE)
test_full <- test_full %>%
  mutate(normalized_spin_efficiency = 
           100*(spin_efficiency - min_spin_efficiency) / (max_spin_efficiency - min_spin_efficiency)) 
rm(max_spin_efficiency, min_spin_efficiency)

fullseason_test$spin_axis_rad <- fullseason_test$spin_axis * pi / 180
fullseason_test <- fullseason_test %>%
  mutate(TSM = release_spin_rate * sin(spin_axis_rad),
         G = release_spin_rate * cos(spin_axis_rad))
fullseason_test <- fullseason_test %>%
  mutate(spin_efficiency = (TSM / release_spin_rate) * 100)
min_spin_efficiency <- min(fullseason_test$spin_efficiency, na.rm = TRUE)
max_spin_efficiency <- max(fullseason_test$spin_efficiency, na.rm = TRUE)
fullseason_test <- fullseason_test %>%
  mutate(normalized_spin_efficiency = 
           100*(spin_efficiency - min_spin_efficiency) / (max_spin_efficiency - min_spin_efficiency)) 
rm(max_spin_efficiency, min_spin_efficiency)

#Calculating Vertical Approach Angle Per Pitch
test_full$vy_f <- -sqrt(test_full$vy0^2 - (2*test_full$ay*(50-(17/12))))
test_full$t <- (test_full$vy_f - test_full$vy0)/test_full$ay
test_full$vz_f <- test_full$vz0 + (test_full$az*test_full$t)
test_full$VAA <- -atan(test_full$vz_f/test_full$vy_f)*(180/pi)

fullseason_test$vy_f <- -sqrt(fullseason_test$vy0^2 - (2*fullseason_test$ay*(50-(17/12))))
fullseason_test$t <- (fullseason_test$vy_f - fullseason_test$vy0)/fullseason_test$ay
fullseason_test$vz_f <- fullseason_test$vz0 + (fullseason_test$az*fullseason_test$t)
fullseason_test$VAA <- -atan(fullseason_test$vz_f/fullseason_test$vy_f)*(180/pi)

#Paring Down Master Data to Key Specs
test_specs <- subset(test_full, select = c('player_name', 'p_throws',
                                           'description', 'pitch_type', 
                                           'release_speed', 'pfx_z', 'pfx_x', 
                                           'release_pos_z','release_extension', 
                                           'release_spin_rate', 
                                           'normalized_spin_efficiency',
                                           'VAA'))

season_specs <- subset(fullseason_test, select = c('player_name', 'p_throws',
                                           'description', 'pitch_type', 
                                           'release_speed', 'pfx_z', 'pfx_x', 
                                           'release_pos_z','release_extension', 
                                           'release_spin_rate', 
                                           'normalized_spin_efficiency',
                                           'VAA'))

#RHFA Model--------------------------------------------------------------------
#Train Data Prepping
RHFAtrain_selectP <- subset(train_specs, pitch_type %in% c('FF','FC','SI'))
RHFAtrain_selectH <- subset(RHFAtrain_selectP, p_throws %in% c('R'))
RHFAtrain <- na.omit(subset(RHFAtrain_selectH, description %in% c('swinging_strike',
                                                                  'swinging_strike_blocked',
                                                                  'hit into play',
                                                                  'foul', 'foul_tip',
                                                                  'foul_bunt', 'ball')))
rm(RHFAtrain_selectH, RHFAtrain_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
RHFAtrain <- RHFAtrain %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHFAtrain$Whiffrate <- 100*(RHFAtrain$Twhiffs/RHFAtrain$Tswings)
RHFAtrain <- na.omit(RHFAtrain)
RHFAtrain <- subset(RHFAtrain, Tswings>25)

#Daily Test Data Prepping
RHFAtest_selectP <- subset(test_specs, pitch_type %in% c('FF','FC','SI'))
RHFAtest_selectH <- subset(RHFAtest_selectP, p_throws %in% c('R'))
RHFAday <- na.omit(subset(RHFAtest_selectH, description %in% c('swinging_strike',
                                                                'swinging_strike_blocked',
                                                                'hit into play',
                                                                'foul', 'foul_tip',
                                                                'foul_bunt', 'ball')))
rm(RHFAtest_selectH, RHFAtest_selectP)

#Season Test Data Prepping
RHFAseason_selectP <- subset(season_specs, pitch_type %in% c('FF','FC','SI'))
RHFAseason_selectH <- subset(RHFAseason_selectP, p_throws %in% c('R'))
RHFAseason <- na.omit(subset(RHFAseason_selectH, description %in% c('swinging_strike',
                                                                    'swinging_strike_blocked',
                                                                    'hit into play',
                                                                    'foul', 'foul_tip',
                                                                    'foul_bunt', 'ball')))
rm(RHFAseason_selectH, RHFAseason_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
RHFAday <- RHFAday %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHFAday$Whiffrate <- 100*(RHFAday$Twhiffs/RHFAday$Tswings)
RHFAday <- na.omit(RHFAday)

#Aggregating Season Key Specs & Calculating Whiff Rate
RHFAseason <- RHFAseason %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHFAseason$Whiffrate <- 100*(RHFAseason$Twhiffs/RHFAseason$Tswings)
RHFAseason <- na.omit(RHFAseason)

#Daily Model Building
RHFAtrain_x <- data.matrix(subset(RHFAtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
RHFAtrain_y = data.matrix(subset(RHFAtrain, select = c(Whiffrate)))

RHFAtest_x <- data.matrix(subset(RHFAday, select = -c(player_name,
                                                       Twhiffs, Tswings,
                                                       Whiffrate, pitch_type)))
RHFAtest_y <- data.matrix(subset(RHFAday, select = c(Whiffrate)))

RHFAxgb_train <- xgb.DMatrix(data = RHFAtrain_x, label = RHFAtrain_y)
RHFAxgb_test = xgb.DMatrix(data = RHFAtest_x, label = RHFAtest_y)

RHFAwatchlist = list(train = RHFAxgb_train, test = RHFAxgb_test)
RHFAmodel = xgb.train(data = RHFAxgb_train, max.depth = 4, 
                      watchlist = RHFAwatchlist, nrounds = 6)

RHFAxgboosted = xgboost(data = RHFAxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
RHFAday$pWhiffrate <- predict(RHFAxgboosted, RHFAxgb_test)

rm(RHFAtest_x, RHFAtest_y, RHFAtrain_x, RHFAtrain_y, RHFAwatchlist, 
   RHFAxgboosted, RHFAxgb_test, RHFAxgb_train)

RHFAday$Stuff <- 100*(RHFAday$pWhiffrate/mean(RHFAday$pWhiffrate))

#Season Model Building
RHFAtrain_x <- data.matrix(subset(RHFAtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
RHFAtrain_y = data.matrix(subset(RHFAtrain, select = c(Whiffrate)))

RHFAtest_x <- data.matrix(subset(RHFAseason, select = -c(player_name,
                                                       Twhiffs, Tswings,
                                                       Whiffrate, pitch_type)))
RHFAtest_y <- data.matrix(subset(RHFAseason, select = c(Whiffrate)))

RHFAxgb_train <- xgb.DMatrix(data = RHFAtrain_x, label = RHFAtrain_y)
RHFAxgb_test = xgb.DMatrix(data = RHFAtest_x, label = RHFAtest_y)

RHFAwatchlist = list(train = RHFAxgb_train, test = RHFAxgb_test)
RHFAmodel = xgb.train(data = RHFAxgb_train, max.depth = 4, 
                      watchlist = RHFAwatchlist, nrounds = 6)

RHFAxgboosted = xgboost(data = RHFAxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
RHFAseason$pWhiffrate <- predict(RHFAxgboosted, RHFAxgb_test)

rm(RHFAtest_x, RHFAtest_y, RHFAtrain, RHFAtrain_x, RHFAtrain_y, RHFAwatchlist, 
   RHFAxgboosted, RHFAxgb_test, RHFAxgb_train)

RHFAseason$Stuff <- 100*(RHFAseason$pWhiffrate/mean(RHFAseason$pWhiffrate))

#LHFA Model--------------------------------------------------------------------
#Train Data Prepping
LHFAtrain_selectP <- subset(train_specs, pitch_type %in% c('FF','FC','SI'))
LHFAtrain_selectH <- subset(LHFAtrain_selectP, p_throws %in% c('L'))
LHFAtrain <- na.omit(subset(LHFAtrain_selectH, description %in% c('swinging_strike',
                                                                  'swinging_strike_blocked',
                                                                  'hit into play',
                                                                  'foul', 'foul_tip',
                                                                  'foul_bunt', 'ball')))
rm(LHFAtrain_selectH, LHFAtrain_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
LHFAtrain <- LHFAtrain %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHFAtrain$Whiffrate <- 100*(LHFAtrain$Twhiffs/LHFAtrain$Tswings)
LHFAtrain <- na.omit(LHFAtrain)
LHFAtrain <- subset(LHFAtrain, Tswings>25)

#Daily Test Data Prepping
LHFAtest_selectP <- subset(test_specs, pitch_type %in% c('FF','FC','SI'))
LHFAtest_selectH <- subset(LHFAtest_selectP, p_throws %in% c('L'))
LHFAday <- na.omit(subset(LHFAtest_selectH, description %in% c('swinging_strike',
                                                               'swinging_strike_blocked',
                                                               'hit into play',
                                                               'foul', 'foul_tip',
                                                               'foul_bunt', 'ball')))
rm(LHFAtest_selectH, LHFAtest_selectP)

#Season Test Data Prepping
LHFAseason_selectP <- subset(season_specs, pitch_type %in% c('FF','FC','SI'))
LHFAseason_selectH <- subset(LHFAseason_selectP, p_throws %in% c('L'))
LHFAseason <- na.omit(subset(LHFAseason_selectH, description %in% c('swinging_strike',
                                                                    'swinging_strike_blocked',
                                                                    'hit into play',
                                                                    'foul', 'foul_tip',
                                                                    'foul_bunt', 'ball')))
rm(LHFAseason_selectH, LHFAseason_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
LHFAday <- LHFAday %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHFAday$Whiffrate <- 100*(LHFAday$Twhiffs/LHFAday$Tswings)
LHFAday <- na.omit(LHFAday)

#Aggregating Season Key Specs & Calculating Whiff Rate
LHFAseason <- LHFAseason %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHFAseason$Whiffrate <- 100*(LHFAseason$Twhiffs/LHFAseason$Tswings)
LHFAseason <- na.omit(LHFAseason)

#Daily Model Building
LHFAtrain_x <- data.matrix(subset(LHFAtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
LHFAtrain_y = data.matrix(subset(LHFAtrain, select = c(Whiffrate)))

LHFAtest_x <- data.matrix(subset(LHFAday, select = -c(player_name,
                                                      Twhiffs, Tswings,
                                                      Whiffrate, pitch_type)))
LHFAtest_y <- data.matrix(subset(LHFAday, select = c(Whiffrate)))

LHFAxgb_train <- xgb.DMatrix(data = LHFAtrain_x, label = LHFAtrain_y)
LHFAxgb_test = xgb.DMatrix(data = LHFAtest_x, label = LHFAtest_y)

LHFAwatchlist = list(train = LHFAxgb_train, test = LHFAxgb_test)
LHFAmodel = xgb.train(data = LHFAxgb_train, max.depth = 4, 
                      watchlist = LHFAwatchlist, nrounds = 6)

LHFAxgboosted = xgboost(data = LHFAxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
LHFAday$pWhiffrate <- predict(LHFAxgboosted, LHFAxgb_test)

rm(LHFAtest_x, LHFAtest_y, LHFAtrain_x, LHFAtrain_y, LHFAwatchlist, 
   LHFAxgboosted, LHFAxgb_test, LHFAxgb_train)

LHFAday$Stuff <- 100*(LHFAday$pWhiffrate/mean(LHFAday$pWhiffrate))

#Season Model Building
LHFAtrain_x <- data.matrix(subset(LHFAtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
LHFAtrain_y = data.matrix(subset(LHFAtrain, select = c(Whiffrate)))

LHFAtest_x <- data.matrix(subset(LHFAseason, select = -c(player_name,
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type)))
LHFAtest_y <- data.matrix(subset(LHFAseason, select = c(Whiffrate)))

LHFAxgb_train <- xgb.DMatrix(data = LHFAtrain_x, label = LHFAtrain_y)
LHFAxgb_test = xgb.DMatrix(data = LHFAtest_x, label = LHFAtest_y)

LHFAwatchlist = list(train = LHFAxgb_train, test = LHFAxgb_test)
LHFAmodel = xgb.train(data = LHFAxgb_train, max.depth = 4, 
                      watchlist = LHFAwatchlist, nrounds = 6)

LHFAxgboosted = xgboost(data = LHFAxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
LHFAseason$pWhiffrate <- predict(LHFAxgboosted, LHFAxgb_test)

rm(LHFAtest_x, LHFAtest_y, LHFAtrain, LHFAtrain_x, LHFAtrain_y, LHFAwatchlist, 
   LHFAxgboosted, LHFAxgb_test, LHFAxgb_train)

LHFAseason$Stuff <- 100*(LHFAseason$pWhiffrate/mean(LHFAseason$pWhiffrate))

#RHCH Model--------------------------------------------------------------------
#Train Data Prepping
RHCHtrain_selectP <- subset(train_specs, pitch_type %in% c('CH','FS'))
RHCHtrain_selectH <- subset(RHCHtrain_selectP, p_throws %in% c('R'))
RHCHtrain <- na.omit(subset(RHCHtrain_selectH, description %in% c('swinging_strike',
                                                                  'swinging_strike_blocked',
                                                                  'hit into play',
                                                                  'foul', 'foul_tip',
                                                                  'foul_bunt', 'ball')))
rm(RHCHtrain_selectH, RHCHtrain_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
RHCHtrain <- RHCHtrain %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            BU = mean(release_spin_rate/release_speed, na.rm = T),
            SpinEff = mean(normalized_spin_efficiency, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHCHtrain$Whiffrate <- 100*(RHCHtrain$Twhiffs/RHCHtrain$Tswings)
RHCHtrain <- na.omit(RHCHtrain)
RHCHtrain <- subset(RHCHtrain, Tswings>25)

#Daily Test Data Prepping
RHCHtest_selectP <- subset(test_specs, pitch_type %in% c('CH','FS'))
RHCHtest_selectH <- subset(RHCHtest_selectP, p_throws %in% c('R'))
RHCHday <- na.omit(subset(RHCHtest_selectH, description %in% c('swinging_strike',
                                                               'swinging_strike_blocked',
                                                               'hit into play',
                                                               'foul', 'foul_tip',
                                                               'foul_bunt', 'ball')))
rm(RHCHtest_selectH, RHCHtest_selectP)

#Season Test Data Prepping
RHCHseason_selectP <- subset(season_specs, pitch_type %in% c('CH','FS'))
RHCHseason_selectH <- subset(RHCHseason_selectP, p_throws %in% c('R'))
RHCHseason <- na.omit(subset(RHCHseason_selectH, description %in% c('swinging_strike',
                                                                    'swinging_strike_blocked',
                                                                    'hit into play',
                                                                    'foul', 'foul_tip',
                                                                    'foul_bunt', 'ball')))
rm(RHCHseason_selectH, RHCHseason_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
RHCHday <- RHCHday %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            BU = mean(release_spin_rate/release_speed, na.rm = T),
            SpinEff = mean(normalized_spin_efficiency, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHCHday$Whiffrate <- 100*(RHCHday$Twhiffs/RHCHday$Tswings)
RHCHday <- na.omit(RHCHday)

#Aggregating Season Key Specs & Calculating Whiff Rate
RHCHseason <- RHCHseason %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            BU = mean(release_spin_rate/release_speed, na.rm = T),
            SpinEff = mean(normalized_spin_efficiency, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHCHseason$Whiffrate <- 100*(RHCHseason$Twhiffs/RHCHseason$Tswings)
RHCHseason <- na.omit(RHCHseason)

#Daily Model Building
RHCHtrain_x <- data.matrix(subset(RHCHtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
RHCHtrain_y = data.matrix(subset(RHCHtrain, select = c(Whiffrate)))

RHCHtest_x <- data.matrix(subset(RHCHday, select = -c(player_name,
                                                      Twhiffs, Tswings,
                                                      Whiffrate, pitch_type)))
RHCHtest_y <- data.matrix(subset(RHCHday, select = c(Whiffrate)))

RHCHxgb_train <- xgb.DMatrix(data = RHCHtrain_x, label = RHCHtrain_y)
RHCHxgb_test = xgb.DMatrix(data = RHCHtest_x, label = RHCHtest_y)

RHCHwatchlist = list(train = RHCHxgb_train, test = RHCHxgb_test)
RHCHmodel = xgb.train(data = RHCHxgb_train, max.depth = 4, 
                      watchlist = RHCHwatchlist, nrounds = 6)

RHCHxgboosted = xgboost(data = RHCHxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
RHCHday$pWhiffrate <- predict(RHCHxgboosted, RHCHxgb_test)

rm(RHCHtest_x, RHCHtest_y, RHCHtrain_x, RHCHtrain_y, RHCHwatchlist, 
   RHCHxgboosted, RHCHxgb_test, RHCHxgb_train)

RHCHday$Stuff <- 100*(RHCHday$pWhiffrate/mean(RHCHday$pWhiffrate))

#Season Model Building
RHCHtrain_x <- data.matrix(subset(RHCHtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
RHCHtrain_y = data.matrix(subset(RHCHtrain, select = c(Whiffrate)))

RHCHtest_x <- data.matrix(subset(RHCHseason, select = -c(player_name,
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type)))
RHCHtest_y <- data.matrix(subset(RHCHseason, select = c(Whiffrate)))

RHCHxgb_train <- xgb.DMatrix(data = RHCHtrain_x, label = RHCHtrain_y)
RHCHxgb_test = xgb.DMatrix(data = RHCHtest_x, label = RHCHtest_y)

RHCHwatchlist = list(train = RHCHxgb_train, test = RHCHxgb_test)
RHCHmodel = xgb.train(data = RHCHxgb_train, max.depth = 4, 
                      watchlist = RHCHwatchlist, nrounds = 6)

RHCHxgboosted = xgboost(data = RHCHxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
RHCHseason$pWhiffrate <- predict(RHCHxgboosted, RHCHxgb_test)

rm(RHCHtest_x, RHCHtest_y, RHCHtrain, RHCHtrain_x, RHCHtrain_y, RHCHwatchlist, 
   RHCHxgboosted, RHCHxgb_test, RHCHxgb_train)

RHCHseason$Stuff <- 100*(RHCHseason$pWhiffrate/mean(RHCHseason$pWhiffrate))

#LHCH Model--------------------------------------------------------------------
#Train Data Prepping
LHCHtrain_selectP <- subset(train_specs, pitch_type %in% c('CH','FS'))
LHCHtrain_selectH <- subset(LHCHtrain_selectP, p_throws %in% c('L'))
LHCHtrain <- na.omit(subset(LHCHtrain_selectH, description %in% c('swinging_strike',
                                                                  'swinging_strike_blocked',
                                                                  'hit into play',
                                                                  'foul', 'foul_tip',
                                                                  'foul_bunt', 'ball')))
rm(LHCHtrain_selectH, LHCHtrain_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
LHCHtrain <- LHCHtrain %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            BU = mean(release_spin_rate/release_speed, na.rm = T),
            SpinEff = mean(normalized_spin_efficiency, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHCHtrain$Whiffrate <- 100*(LHCHtrain$Twhiffs/LHCHtrain$Tswings)
LHCHtrain <- na.omit(LHCHtrain)
LHCHtrain <- subset(LHCHtrain, Tswings>25)

#Daily Test Data Prepping
LHCHtest_selectP <- subset(test_specs, pitch_type %in% c('CH','FS'))
LHCHtest_selectH <- subset(LHCHtest_selectP, p_throws %in% c('L'))
LHCHday <- na.omit(subset(LHCHtest_selectH, description %in% c('swinging_strike',
                                                               'swinging_strike_blocked',
                                                               'hit into play',
                                                               'foul', 'foul_tip',
                                                               'foul_bunt', 'ball')))
rm(LHCHtest_selectH, LHCHtest_selectP)

#Season Test Data Prepping
LHCHseason_selectP <- subset(season_specs, pitch_type %in% c('CH','FS'))
LHCHseason_selectH <- subset(LHCHseason_selectP, p_throws %in% c('L'))
LHCHseason <- na.omit(subset(LHCHseason_selectH, description %in% c('swinging_strike',
                                                                    'swinging_strike_blocked',
                                                                    'hit into play',
                                                                    'foul', 'foul_tip',
                                                                    'foul_bunt', 'ball')))
rm(LHCHseason_selectH, LHCHseason_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
LHCHday <- LHCHday %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            BU = mean(release_spin_rate/release_speed, na.rm = T),
            SpinEff = mean(normalized_spin_efficiency, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHCHday$Whiffrate <- 100*(LHCHday$Twhiffs/LHCHday$Tswings)
LHCHday <- na.omit(LHCHday)

#Aggregating Season Key Specs & Calculating Whiff Rate
LHCHseason <- LHCHseason %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            BU = mean(release_spin_rate/release_speed, na.rm = T),
            SpinEff = mean(normalized_spin_efficiency, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHCHseason$Whiffrate <- 100*(LHCHseason$Twhiffs/LHCHseason$Tswings)
LHCHseason <- na.omit(LHCHseason)

#Daily Model Building
LHCHtrain_x <- data.matrix(subset(LHCHtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
LHCHtrain_y = data.matrix(subset(LHCHtrain, select = c(Whiffrate)))

LHCHtest_x <- data.matrix(subset(LHCHday, select = -c(player_name,
                                                      Twhiffs, Tswings,
                                                      Whiffrate, pitch_type)))
LHCHtest_y <- data.matrix(subset(LHCHday, select = c(Whiffrate)))

LHCHxgb_train <- xgb.DMatrix(data = LHCHtrain_x, label = LHCHtrain_y)
LHCHxgb_test = xgb.DMatrix(data = LHCHtest_x, label = LHCHtest_y)

LHCHwatchlist = list(train = LHCHxgb_train, test = LHCHxgb_test)
LHCHmodel = xgb.train(data = LHCHxgb_train, max.depth = 4, 
                      watchlist = LHCHwatchlist, nrounds = 6)

LHCHxgboosted = xgboost(data = LHCHxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
LHCHday$pWhiffrate <- predict(LHCHxgboosted, LHCHxgb_test)

rm(LHCHtest_x, LHCHtest_y, LHCHtrain_x, LHCHtrain_y, LHCHwatchlist, 
   LHCHxgboosted, LHCHxgb_test, LHCHxgb_train)

LHCHday$Stuff <- 100*(LHCHday$pWhiffrate/mean(LHCHday$pWhiffrate))

#Season Model Building
LHCHtrain_x <- data.matrix(subset(LHCHtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
LHCHtrain_y = data.matrix(subset(LHCHtrain, select = c(Whiffrate)))

LHCHtest_x <- data.matrix(subset(LHCHseason, select = -c(player_name,
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type)))
LHCHtest_y <- data.matrix(subset(LHCHseason, select = c(Whiffrate)))

LHCHxgb_train <- xgb.DMatrix(data = LHCHtrain_x, label = LHCHtrain_y)
LHCHxgb_test = xgb.DMatrix(data = LHCHtest_x, label = LHCHtest_y)

LHCHwatchlist = list(train = LHCHxgb_train, test = LHCHxgb_test)
LHCHmodel = xgb.train(data = LHCHxgb_train, max.depth = 4, 
                      watchlist = LHCHwatchlist, nrounds = 6)

LHCHxgboosted = xgboost(data = LHCHxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
LHCHseason$pWhiffrate <- predict(LHCHxgboosted, LHCHxgb_test)

rm(LHCHtest_x, LHCHtest_y, LHCHtrain, LHCHtrain_x, LHCHtrain_y, LHCHwatchlist, 
   LHCHxgboosted, LHCHxgb_test, LHCHxgb_train)

LHCHseason$Stuff <- 100*(LHCHseason$pWhiffrate/mean(LHCHseason$pWhiffrate))

#RHSL Model--------------------------------------------------------------------
#Train Data Prepping
RHSLtrain_selectP <- subset(train_specs, pitch_type %in% c('SL','ST'))
RHSLtrain_selectH <- subset(RHSLtrain_selectP, p_throws %in% c('R'))
RHSLtrain <- na.omit(subset(RHSLtrain_selectH, description %in% c('swinging_strike',
                                                                  'swinging_strike_blocked',
                                                                  'hit into play',
                                                                  'foul', 'foul_tip',
                                                                  'foul_bunt', 'ball')))
rm(RHSLtrain_selectH, RHSLtrain_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
RHSLtrain <- RHSLtrain %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHSLtrain$Whiffrate <- 100*(RHSLtrain$Twhiffs/RHSLtrain$Tswings)
RHSLtrain <- na.omit(RHSLtrain)
RHSLtrain <- subset(RHSLtrain, Tswings>25)

#Daily Test Data Prepping
RHSLtest_selectP <- subset(test_specs, pitch_type %in% c('SL','ST'))
RHSLtest_selectH <- subset(RHSLtest_selectP, p_throws %in% c('R'))
RHSLday <- na.omit(subset(RHSLtest_selectH, description %in% c('swinging_strike',
                                                               'swinging_strike_blocked',
                                                               'hit into play',
                                                               'foul', 'foul_tip',
                                                               'foul_bunt', 'ball')))
rm(RHSLtest_selectH, RHSLtest_selectP)

#Season Test Data Prepping
RHSLseason_selectP <- subset(season_specs, pitch_type %in% c('SL','SL'))
RHSLseason_selectH <- subset(RHSLseason_selectP, p_throws %in% c('R'))
RHSLseason <- na.omit(subset(RHSLseason_selectH, description %in% c('swinging_strike',
                                                                    'swinging_strike_blocked',
                                                                    'hit into play',
                                                                    'foul', 'foul_tip',
                                                                    'foul_bunt', 'ball')))
rm(RHSLseason_selectH, RHSLseason_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
RHSLday <- RHSLday %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHSLday$Whiffrate <- 100*(RHSLday$Twhiffs/RHSLday$Tswings)
RHSLday <- na.omit(RHSLday)

#Aggregating Season Key Specs & Calculating Whiff Rate
RHSLseason <- RHSLseason %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHSLseason$Whiffrate <- 100*(RHSLseason$Twhiffs/RHSLseason$Tswings)
RHSLseason <- na.omit(RHSLseason)

#Daily Model Building
RHSLtrain_x <- data.matrix(subset(RHSLtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
RHSLtrain_y = data.matrix(subset(RHSLtrain, select = c(Whiffrate)))

RHSLtest_x <- data.matrix(subset(RHSLday, select = -c(player_name,
                                                      Twhiffs, Tswings,
                                                      Whiffrate, pitch_type)))
RHSLtest_y <- data.matrix(subset(RHSLday, select = c(Whiffrate)))

RHSLxgb_train <- xgb.DMatrix(data = RHSLtrain_x, label = RHSLtrain_y)
RHSLxgb_test = xgb.DMatrix(data = RHSLtest_x, label = RHSLtest_y)

RHSLwatchlist = list(train = RHSLxgb_train, test = RHSLxgb_test)
RHSLmodel = xgb.train(data = RHSLxgb_train, max.depth = 4, 
                      watchlist = RHSLwatchlist, nrounds = 6)

RHSLxgboosted = xgboost(data = RHSLxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
RHSLday$pWhiffrate <- predict(RHSLxgboosted, RHSLxgb_test)

rm(RHSLtest_x, RHSLtest_y, RHSLtrain_x, RHSLtrain_y, RHSLwatchlist, 
   RHSLxgboosted, RHSLxgb_test, RHSLxgb_train)

RHSLday$Stuff <- 100*(RHSLday$pWhiffrate/mean(RHSLday$pWhiffrate))

#Season Model Building
RHSLtrain_x <- data.matrix(subset(RHSLtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
RHSLtrain_y = data.matrix(subset(RHSLtrain, select = c(Whiffrate)))

RHSLtest_x <- data.matrix(subset(RHSLseason, select = -c(player_name,
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type)))
RHSLtest_y <- data.matrix(subset(RHSLseason, select = c(Whiffrate)))

RHSLxgb_train <- xgb.DMatrix(data = RHSLtrain_x, label = RHSLtrain_y)
RHSLxgb_test = xgb.DMatrix(data = RHSLtest_x, label = RHSLtest_y)

RHSLwatchlist = list(train = RHSLxgb_train, test = RHSLxgb_test)
RHSLmodel = xgb.train(data = RHSLxgb_train, max.depth = 4, 
                      watchlist = RHSLwatchlist, nrounds = 6)

RHSLxgboosted = xgboost(data = RHSLxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
RHSLseason$pWhiffrate <- predict(RHSLxgboosted, RHSLxgb_test)

rm(RHSLtest_x, RHSLtest_y, RHSLtrain, RHSLtrain_x, RHSLtrain_y, RHSLwatchlist, 
   RHSLxgboosted, RHSLxgb_test, RHSLxgb_train)

RHSLseason$Stuff <- 100*(RHSLseason$pWhiffrate/mean(RHSLseason$pWhiffrate))

#LHSL Model--------------------------------------------------------------------
#Train Data Prepping
LHSLtrain_selectP <- subset(train_specs, pitch_type %in% c('SL','ST'))
LHSLtrain_selectH <- subset(LHSLtrain_selectP, p_throws %in% c('L'))
LHSLtrain <- na.omit(subset(LHSLtrain_selectH, description %in% c('swinging_strike',
                                                                  'swinging_strike_blocked',
                                                                  'hit into play',
                                                                  'foul', 'foul_tip',
                                                                  'foul_bunt', 'ball')))
rm(LHSLtrain_selectH, LHSLtrain_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
LHSLtrain <- LHSLtrain %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHSLtrain$Whiffrate <- 100*(LHSLtrain$Twhiffs/LHSLtrain$Tswings)
LHSLtrain <- na.omit(LHSLtrain)
LHSLtrain <- subset(LHSLtrain, Tswings>25)

#Daily Test Data Prepping
LHSLtest_selectP <- subset(test_specs, pitch_type %in% c('SL','ST'))
LHSLtest_selectH <- subset(LHSLtest_selectP, p_throws %in% c('L'))
LHSLday <- na.omit(subset(LHSLtest_selectH, description %in% c('swinging_strike',
                                                               'swinging_strike_blocked',
                                                               'hit into play',
                                                               'foul', 'foul_tip',
                                                               'foul_bunt', 'ball')))
rm(LHSLtest_selectH, LHSLtest_selectP)

#Season Test Data Prepping
LHSLseason_selectP <- subset(season_specs, pitch_type %in% c('SL','ST'))
LHSLseason_selectH <- subset(LHSLseason_selectP, p_throws %in% c('L'))
LHSLseason <- na.omit(subset(LHSLseason_selectH, description %in% c('swinging_strike',
                                                                    'swinging_strike_blocked',
                                                                    'hit into play',
                                                                    'foul', 'foul_tip',
                                                                    'foul_bunt', 'ball')))
rm(LHSLseason_selectH, LHSLseason_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
LHSLday <- LHSLday %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHSLday$Whiffrate <- 100*(LHSLday$Twhiffs/LHSLday$Tswings)
LHSLday <- na.omit(LHSLday)

#Aggregating Season Key Specs & Calculating Whiff Rate
LHSLseason <- LHSLseason %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHSLseason$Whiffrate <- 100*(LHSLseason$Twhiffs/LHSLseason$Tswings)
LHSLseason <- na.omit(LHSLseason)

#Daily Model Building
LHSLtrain_x <- data.matrix(subset(LHSLtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
LHSLtrain_y = data.matrix(subset(LHSLtrain, select = c(Whiffrate)))

LHSLtest_x <- data.matrix(subset(LHSLday, select = -c(player_name,
                                                      Twhiffs, Tswings,
                                                      Whiffrate, pitch_type)))
LHSLtest_y <- data.matrix(subset(LHSLday, select = c(Whiffrate)))

LHSLxgb_train <- xgb.DMatrix(data = LHSLtrain_x, label = LHSLtrain_y)
LHSLxgb_test = xgb.DMatrix(data = LHSLtest_x, label = LHSLtest_y)

LHSLwatchlist = list(train = LHSLxgb_train, test = LHSLxgb_test)
LHSLmodel = xgb.train(data = LHSLxgb_train, max.depth = 4, 
                      watchlist = LHSLwatchlist, nrounds = 6)

LHSLxgboosted = xgboost(data = LHSLxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
LHSLday$pWhiffrate <- predict(LHSLxgboosted, LHSLxgb_test)

rm(LHSLtest_x, LHSLtest_y, LHSLtrain_x, LHSLtrain_y, LHSLwatchlist, 
   LHSLxgboosted, LHSLxgb_test, LHSLxgb_train)

LHSLday$Stuff <- 100*(LHSLday$pWhiffrate/mean(LHSLday$pWhiffrate))

#Season Model Building
LHSLtrain_x <- data.matrix(subset(LHSLtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
LHSLtrain_y = data.matrix(subset(LHSLtrain, select = c(Whiffrate)))

LHSLtest_x <- data.matrix(subset(LHSLseason, select = -c(player_name,
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type)))
LHSLtest_y <- data.matrix(subset(LHSLseason, select = c(Whiffrate)))

LHSLxgb_train <- xgb.DMatrix(data = LHSLtrain_x, label = LHSLtrain_y)
LHSLxgb_test = xgb.DMatrix(data = LHSLtest_x, label = LHSLtest_y)

LHSLwatchlist = list(train = LHSLxgb_train, test = LHSLxgb_test)
LHSLmodel = xgb.train(data = LHSLxgb_train, max.depth = 4, 
                      watchlist = LHSLwatchlist, nrounds = 6)

LHSLxgboosted = xgboost(data = LHSLxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
LHSLseason$pWhiffrate <- predict(LHSLxgboosted, LHSLxgb_test)

rm(LHSLtest_x, LHSLtest_y, LHSLtrain, LHSLtrain_x, LHSLtrain_y, LHSLwatchlist, 
   LHSLxgboosted, LHSLxgb_test, LHSLxgb_train)

LHSLseason$Stuff <- 100*(LHSLseason$pWhiffrate/mean(LHSLseason$pWhiffrate))

#RHCU--------------------------------------------------------------------------
#Train Data Prepping
RHCUtrain_selectP <- subset(train_specs, pitch_type %in% c('CU','KC'))
RHCUtrain_selectH <- subset(RHCUtrain_selectP, p_throws %in% c('R'))
RHCUtrain <- na.omit(subset(RHCUtrain_selectH, description %in% c('swinging_strike',
                                                                  'swinging_strike_blocked',
                                                                  'hit into play',
                                                                  'foul', 'foul_tip',
                                                                  'foul_bunt', 'ball')))
rm(RHCUtrain_selectH, RHCUtrain_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
RHCUtrain <- RHCUtrain %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHCUtrain$Whiffrate <- 100*(RHCUtrain$Twhiffs/RHCUtrain$Tswings)
RHCUtrain <- na.omit(RHCUtrain)
RHCUtrain <- subset(RHCUtrain, Tswings>25)

#Daily Test Data Prepping
RHCUtest_selectP <- subset(test_specs, pitch_type %in% c('CU','KC'))
RHCUtest_selectH <- subset(RHCUtest_selectP, p_throws %in% c('R'))
RHCUday <- na.omit(subset(RHCUtest_selectH, description %in% c('swinging_strike',
                                                               'swinging_strike_blocked',
                                                               'hit into play',
                                                               'foul', 'foul_tip',
                                                               'foul_bunt', 'ball')))
rm(RHCUtest_selectH, RHCUtest_selectP)

#Season Test Data Prepping
RHCUseason_selectP <- subset(season_specs, pitch_type %in% c('CU','KC'))
RHCUseason_selectH <- subset(RHCUseason_selectP, p_throws %in% c('R'))
RHCUseason <- na.omit(subset(RHCUseason_selectH, description %in% c('swinging_strike',
                                                                    'swinging_strike_blocked',
                                                                    'hit into play',
                                                                    'foul', 'foul_tip',
                                                                    'foul_bunt', 'ball')))
rm(RHCUseason_selectH, RHCUseason_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
RHCUday <- RHCUday %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHCUday$Whiffrate <- 100*(RHCUday$Twhiffs/RHCUday$Tswings)
RHCUday <- na.omit(RHCUday)

#Aggregating Season Key Specs & Calculating Whiff Rate
RHCUseason <- RHCUseason %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
RHCUseason$Whiffrate <- 100*(RHCUseason$Twhiffs/RHCUseason$Tswings)
RHCUseason <- na.omit(RHCUseason)

#Daily Model Building
RHCUtrain_x <- data.matrix(subset(RHCUtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
RHCUtrain_y = data.matrix(subset(RHCUtrain, select = c(Whiffrate)))

RHCUtest_x <- data.matrix(subset(RHCUday, select = -c(player_name,
                                                      Twhiffs, Tswings,
                                                      Whiffrate, pitch_type)))
RHCUtest_y <- data.matrix(subset(RHCUday, select = c(Whiffrate)))

RHCUxgb_train <- xgb.DMatrix(data = RHCUtrain_x, label = RHCUtrain_y)
RHCUxgb_test = xgb.DMatrix(data = RHCUtest_x, label = RHCUtest_y)

RHCUwatchlist = list(train = RHCUxgb_train, test = RHCUxgb_test)
RHCUmodel = xgb.train(data = RHCUxgb_train, max.depth = 4, 
                      watchlist = RHCUwatchlist, nrounds = 6)

RHCUxgboosted = xgboost(data = RHCUxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
RHCUday$pWhiffrate <- predict(RHCUxgboosted, RHCUxgb_test)

rm(RHCUtest_x, RHCUtest_y, RHCUtrain_x, RHCUtrain_y, RHCUwatchlist, 
   RHCUxgboosted, RHCUxgb_test, RHCUxgb_train)

RHCUday$Stuff <- 100*(RHCUday$pWhiffrate/mean(RHCUday$pWhiffrate))

#Season Model Building
RHCUtrain_x <- data.matrix(subset(RHCUtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
RHCUtrain_y = data.matrix(subset(RHCUtrain, select = c(Whiffrate)))

RHCUtest_x <- data.matrix(subset(RHCUseason, select = -c(player_name,
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type)))
RHCUtest_y <- data.matrix(subset(RHCUseason, select = c(Whiffrate)))

RHCUxgb_train <- xgb.DMatrix(data = RHCUtrain_x, label = RHCUtrain_y)
RHCUxgb_test = xgb.DMatrix(data = RHCUtest_x, label = RHCUtest_y)

RHCUwatchlist = list(train = RHCUxgb_train, test = RHCUxgb_test)
RHCUmodel = xgb.train(data = RHCUxgb_train, max.depth = 4, 
                      watchlist = RHCUwatchlist, nrounds = 6)

RHCUxgboosted = xgboost(data = RHCUxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
RHCUseason$pWhiffrate <- predict(RHCUxgboosted, RHCUxgb_test)

rm(RHCUtest_x, RHCUtest_y, RHCUtrain, RHCUtrain_x, RHCUtrain_y, RHCUwatchlist, 
   RHCUxgboosted, RHCUxgb_test, RHCUxgb_train)

RHCUseason$Stuff <- 100*(RHCUseason$pWhiffrate/mean(RHCUseason$pWhiffrate))

#LHCU--------------------------------------------------------------------------
#Train Data Prepping
LHCUtrain_selectP <- subset(train_specs, pitch_type %in% c('CU','KC'))
LHCUtrain_selectH <- subset(LHCUtrain_selectP, p_throws %in% c('L'))
LHCUtrain <- na.omit(subset(LHCUtrain_selectH, description %in% c('swinging_strike',
                                                                  'swinging_strike_blocked',
                                                                  'hit into play',
                                                                  'foul', 'foul_tip',
                                                                  'foul_bunt', 'ball')))
rm(LHCUtrain_selectH, LHCUtrain_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
LHCUtrain <- LHCUtrain %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHCUtrain$Whiffrate <- 100*(LHCUtrain$Twhiffs/LHCUtrain$Tswings)
LHCUtrain <- na.omit(LHCUtrain)
LHCUtrain <- subset(LHCUtrain, Tswings>25)

#Daily Test Data Prepping
LHCUtest_selectP <- subset(test_specs, pitch_type %in% c('CU','KC'))
LHCUtest_selectH <- subset(LHCUtest_selectP, p_throws %in% c('L'))
LHCUday <- na.omit(subset(LHCUtest_selectH, description %in% c('swinging_strike',
                                                               'swinging_strike_blocked',
                                                               'hit into play',
                                                               'foul', 'foul_tip',
                                                               'foul_bunt', 'ball')))
rm(LHCUtest_selectH, LHCUtest_selectP)

#Season Test Data Prepping
LHCUseason_selectP <- subset(season_specs, pitch_type %in% c('CU','KC'))
LHCUseason_selectH <- subset(LHCUseason_selectP, p_throws %in% c('L'))
LHCUseason <- na.omit(subset(LHCUseason_selectH, description %in% c('swinging_strike',
                                                                    'swinging_strike_blocked',
                                                                    'hit into play',
                                                                    'foul', 'foul_tip',
                                                                    'foul_bunt', 'ball')))
rm(LHCUseason_selectH, LHCUseason_selectP)

#Aggregating Daily Key Specs & Calculating Whiff Rate
LHCUday <- LHCUday %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHCUday$Whiffrate <- 100*(LHCUday$Twhiffs/LHCUday$Tswings)
LHCUday <- na.omit(LHCUday)

#Aggregating Season Key Specs & Calculating Whiff Rate
LHCUseason <- LHCUseason %>%
  group_by(player_name, pitch_type) %>%
  summarise(Velocity = mean(release_speed, na.rm = T),
            Vmov = 12*mean(pfx_z, na.rm = T),
            Hmov = 12*mean(pfx_x, na.rm = T),
            Spinrate = mean(release_spin_rate, na.rm = T),
            VAA = mean(VAA, na.rm = T),
            Twhiffs = sum(description %in% c('swinging_strike',
                                             'swinging_strike_blocked',
                                             'foul_tip')),
            Tswings = sum(description %in% c('hit_into_play',
                                             'foul', 'foul_bunt',
                                             'foul_tip', 'swinging_strike',
                                             'swinging_strike_blocked')))
LHCUseason$Whiffrate <- 100*(LHCUseason$Twhiffs/LHCUseason$Tswings)
LHCUseason <- na.omit(LHCUseason)

#Daily Model Building
LHCUtrain_x <- data.matrix(subset(LHCUtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
LHCUtrain_y = data.matrix(subset(LHCUtrain, select = c(Whiffrate)))

LHCUtest_x <- data.matrix(subset(LHCUday, select = -c(player_name,
                                                      Twhiffs, Tswings,
                                                      Whiffrate, pitch_type)))
LHCUtest_y <- data.matrix(subset(LHCUday, select = c(Whiffrate)))

LHCUxgb_train <- xgb.DMatrix(data = LHCUtrain_x, label = LHCUtrain_y)
LHCUxgb_test = xgb.DMatrix(data = LHCUtest_x, label = LHCUtest_y)

LHCUwatchlist = list(train = LHCUxgb_train, test = LHCUxgb_test)
LHCUmodel = xgb.train(data = LHCUxgb_train, max.depth = 4, 
                      watchlist = LHCUwatchlist, nrounds = 6)

LHCUxgboosted = xgboost(data = LHCUxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
LHCUday$pWhiffrate <- predict(LHCUxgboosted, LHCUxgb_test)

rm(LHCUtest_x, LHCUtest_y, LHCUtrain_x, LHCUtrain_y, LHCUwatchlist, 
   LHCUxgboosted, LHCUxgb_test, LHCUxgb_train)

LHCUday$Stuff <- 100*(LHCUday$pWhiffrate/mean(LHCUday$pWhiffrate))

#Season Model Building
LHCUtrain_x <- data.matrix(subset(LHCUtrain, select = -c(player_name, 
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type))) 
LHCUtrain_y = data.matrix(subset(LHCUtrain, select = c(Whiffrate)))

LHCUtest_x <- data.matrix(subset(LHCUseason, select = -c(player_name,
                                                         Twhiffs, Tswings,
                                                         Whiffrate, pitch_type)))
LHCUtest_y <- data.matrix(subset(LHCUseason, select = c(Whiffrate)))

LHCUxgb_train <- xgb.DMatrix(data = LHCUtrain_x, label = LHCUtrain_y)
LHCUxgb_test = xgb.DMatrix(data = LHCUtest_x, label = LHCUtest_y)

LHCUwatchlist = list(train = LHCUxgb_train, test = LHCUxgb_test)
LHCUmodel = xgb.train(data = LHCUxgb_train, max.depth = 4, 
                      watchlist = LHCUwatchlist, nrounds = 6)

LHCUxgboosted = xgboost(data = LHCUxgb_train, max.depth = 4, nrounds = 6, 
                        verbose = 0, min_child_weight = 8, eta = 0.8, lambda = 0.1)
LHCUseason$pWhiffrate <- predict(LHCUxgboosted, LHCUxgb_test)

rm(LHCUtest_x, LHCUtest_y, LHCUtrain, LHCUtrain_x, LHCUtrain_y, LHCUwatchlist, 
   LHCUxgboosted, LHCUxgb_test, LHCUxgb_train)

LHCUseason$Stuff <- 100*(LHCUseason$pWhiffrate/mean(LHCUseason$pWhiffrate))

#Stuff Aggregation-------------------------------------------------------------
DayStuff <- rbind(RHFAday, LHFAday, RHCHday, LHCHday, RHSLday, LHSLday, RHCUday, 
                  LHCUday)
DayStuff <- subset(DayStuff, select = c('player_name', 'pitch_type', 'Velocity',
                                        'Vmov','Hmov', 'Whiffrate','pWhiffrate',
                                        'Stuff'))
SeasonStuff <- rbind(RHFAseason, LHFAseason, RHCHseason, LHCHseason, RHSLseason,
                     LHSLseason, RHCUseason, LHCUseason)
SeasonStuff <- subset(SeasonStuff, select = c('player_name', 'pitch_type', 'Velocity',
                                        'Vmov','Hmov', 'Whiffrate','pWhiffrate',
                                        'Stuff'))





