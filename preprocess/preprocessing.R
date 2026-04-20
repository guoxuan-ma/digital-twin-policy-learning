# preprocessing pipeline, using facsimile data as an illustration

load("../data/raw_ehr_fascimile_data.rdata") # load facsimile data


# convert wide EHR data to monthly long format
source("create_data_helpers.R") # load helpers to convert wide EHR data to monthly long format
start_date = as.Date("2020-03-01")
end_date = as.Date("2022-06-30")
demo_var = c("Age.FirstDose", "Gender", "Race", "Visits", "imm_baseline", "windex")
long_ehr = create_long_data_month(wide_ehr, 10, 100, end_date, start_date, demo_var)

# categorization; delete records with unknown gender or unreported race
long_ehr$monthsLastVaxCat = cut(long_ehr$monthsLastVax, breaks = c(-1, -0.1, 1, 3, 6, 12, 100), include.lowest = T)
long_ehr$monthsLastInfCat = cut(long_ehr$monthsLastInf, breaks = c(-1, -0.1, 1, 3, 6, 12, 100), include.lowest = T)
levels(long_ehr$monthsLastVaxCat) = c("none", "0-1", "2-3", "4-6", "7-12", "12+")
levels(long_ehr$monthsLastInfCat) = c("none", "0-1", "2-3", "4-6", "7-12", "12+")
long_ehr$variant = "none"
long_ehr$variant[long_ehr$month >= "2021-07-01"] = "delta"
long_ehr$variant[long_ehr$month >= "2021-12-16"] = "omicron"
long_ehr$variant = factor(long_ehr$variant)
long_ehr$variant = relevel(long_ehr$variant, ref = "none")
long_ehr$inf_next = (long_ehr$reward <= -10) + 0
long_ehr = long_ehr[long_ehr$Gender != "U", ]
long_ehr = long_ehr[long_ehr$Race != "", ]
long_ehr[long_ehr$Race %in% c("American Indian or Alaska Native", "Native Hawaiian and Other Pacific Islander", "Asian"), "Race"] = "Other"

# integrate severe infection into the long EHR monthly data, as severe infection is considered as the endpoint
severe_outcome$DxDate = as.Date(severe_outcome$DxDate)
severe_outcome = severe_outcome[!duplicated(severe_outcome$PatID), ]
severe_outcome = severe_outcome[severe_outcome$DxDate <= "2022-05-31", ]

long_ehr$severe_infection = 0
long_ehr$severe_infection_next = 0
Patid_list = unique(long_ehr$id)
c = 0
for (patid in severe_outcome$PatID) {
  c = c + 1
  cat(c, " ")
  if (patid %in% Patid_list) {
    patid_data = long_ehr[long_ehr$id == patid, ]
    date_difference = severe_outcome[severe_outcome$PatID == patid, "DxDate"] - patid_data$month
    row_idx = which(date_difference == min(date_difference[date_difference > 0]))
    month_severe_infection = patid_data$month[row_idx]
    month_severe_infection_next = patid_data$month[row_idx - 1]
    long_ehr[(long_ehr$id == patid) & (long_ehr$month == month_severe_infection), "severe_infection"] = 1
    long_ehr[(long_ehr$id == patid) & (long_ehr$month == month_severe_infection_next), "severe_infection_next"] = 1
  }
}

# extract relevant columns
long_ehr = long_ehr[, c("id", "action", "Age.FirstDose", "Gender", "Race", "Visits", "imm_baseline", "windex", 
                        "numVax", "variant", "severe_infection_next", "inf_next")]
write.csv(long_ehr, "../data/facsimile_data.csv")