simulate_facsimile_wide_ehr = function(n) {
  numVax = sample(0:4, n, replace = T)
  numInf = sample(0:3, n, replace = T)
  has_severeInf = rep(0, n)
  has_severeInf[numInf > 0] = sample(c(0, 1), sum(numInf > 0), replace = T, prob = c(0.7, 0.3))
  deceased = sample(c(0, 1), n, replace = T, prob = c(0.9, 0.1))
  
  Age.FirstDose = round(runif(n, 0, 90))
  Gender = sample(c("F", "M"), n, replace = T)
  Race = sample(c("African American", "Caucasian", "Other"), n, replace = T)
  Visits = round(runif(n, 0, 100))
  imm_baseline = sample(c(0, 1), n, replace = T, prob = c(0.8, 0.2))
  windex = round(runif(n, 0, 8))
  FirstInf_Date = sample(seq(as.Date("2020-03-12"), as.Date("2022-12-31"), by = "day"), size = n, replace = T)
  FirstInf_Date[numInf == 0] = NA
  SecondInf_Date = FirstInf_Date + runif(n, 30, 360)
  SecondInf_Date[numInf == 1] = NA
  ThirdInf_Date = SecondInf_Date + runif(n, 30, 360)
  ThirdInf_Date[numInf == 2] = NA
  DateVax1 = sample(seq(as.Date("2020-11-04"), as.Date("2021-05-31"), by = "day"), size = n, replace = T)
  DateVax1[numVax == 0] = NA
  DateVax2 = DateVax1 + runif(n, 21, 60)
  DateVax2[numVax == 1] = NA
  DateVax3 = DateVax2 + runif(n, 21, 60)
  DateVax3[numVax == 2] = NA
  DateVax4 = DateVax3 + runif(n, 60, 180)
  DateVax4[numVax == 3] = NA
  
  DeceasedDate = sample(seq(as.Date("2021-01-01"), as.Date("2022-12-31"), by = "day"), size = n, replace = T)
  DeceasedDate[deceased == 0] = NA
  
  facsimile_wide_ehr = data.frame(
    DEID_PatientID = 1:n, 
    Age.FirstDose = Age.FirstDose,
    Gender = Gender,
    Race = Race, 
    Visits = Visits, 
    imm_baseline = imm_baseline,
    windex = windex,
    FirstInf_Date = FirstInf_Date,
    SecondInf_Date = SecondInf_Date,
    ThirdInf_Date = ThirdInf_Date,
    DateVax1 = DateVax1,
    DateVax2 = DateVax2, 
    DateVax3 = DateVax3, 
    DateVax4 = DateVax4, 
    DeceasedDate = DeceasedDate
  )
  facsimile_wide_ehr$FirstInf_Date[facsimile_wide_ehr$FirstInf_Date > facsimile_wide_ehr$DeceasedDate] = NA
  facsimile_wide_ehr$SecondInf_Date[facsimile_wide_ehr$SecondInf_Date > facsimile_wide_ehr$DeceasedDate] = NA
  facsimile_wide_ehr$ThirdInf_Date[facsimile_wide_ehr$ThirdInf_Date > facsimile_wide_ehr$DeceasedDate] = NA
  facsimile_wide_ehr$DateVax1[facsimile_wide_ehr$DateVax1 > facsimile_wide_ehr$DeceasedDate] = NA
  facsimile_wide_ehr$DateVax2[facsimile_wide_ehr$DateVax2 > facsimile_wide_ehr$DeceasedDate] = NA
  facsimile_wide_ehr$DateVax3[facsimile_wide_ehr$DateVax3 > facsimile_wide_ehr$DeceasedDate] = NA
  facsimile_wide_ehr$DateVax4[facsimile_wide_ehr$DateVax4 > facsimile_wide_ehr$DeceasedDate] = NA
  
  severeInf_id = c()
  severeInf_Date = c()
  for (i in 1:n) {
    if (has_severeInf[i] == 1) {
      date = facsimile_wide_ehr[i, c("FirstInf_Date", "SecondInf_Date", "ThirdInf_Date")]
      if (mean(is.na(date)) == 1) next
      severeInf_id = c(severeInf_id, i)
      severeInf_Date = c(severeInf_Date, sample(date[!is.na(date)], size = 1))
    }
  }
  severe_outcome = data.frame(PatID = severeInf_id, DxDate = severeInf_Date)
  
  return(list(facsimile_wide_ehr = facsimile_wide_ehr, severe_outcome = severe_outcome))
} # demographics, clinical variables, vaccination record and infection history are generated only for illustration purposes

set.seed(1)
simulated_facsimile_data = simulate_facsimile_wide_ehr(n = 1000)

wide_ehr = simulated_facsimile_data$facsimile_wide_ehr # wide EHR data
severe_outcome = simulated_facsimile_data$severe_outcome # a standalone dataset of (patient_id, severe_infection) pair

save(wide_ehr, severe_outcome, file = "../data/raw_ehr_fascimile_data.rdata")