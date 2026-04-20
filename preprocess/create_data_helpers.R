month_diff = function(date1, date2) {
  end_date = as.POSIXlt(date1)
  start_date = as.POSIXlt(date2)
  d = 12 * (end_date$year - start_date$year) + (end_date$mon - start_date$mon)
  return(d)
}

create_long_data_month_i = function(rowdata, end_date, start_date, demo_var) {
  demo_info = rowdata[, demo_var]
  inf = rowdata[, c("FirstInf_Date", "SecondInf_Date", "ThirdInf_Date")]
  vax = rowdata[, c("DateVax1", "DateVax2", "DateVax3", "DateVax4")]
  colnames(inf) = c("inf1", "inf2", "inf3")
  colnames(vax) = c("vax1", "vax2", "vax3", "vax4")
  
  terminal_date = min(rowdata$DeceasedDate, end_date, na.rm = T)
  
  event = t(cbind(start_date, inf, vax, terminal_date))
  ordr = order(event)
  ordered_date = event[ordr, ]
  ordered_date = ordered_date[!is.na(ordered_date)]
  ordered_date = as.Date(ordered_date)
  ordered_event = names(ordered_date)
  terminate = which(names(ordered_date) == "terminal_date")
  ordered_date = ordered_date[1:terminate]
  ordered_event = ordered_event[1:terminate]
  
  last_inf_date = start_date
  last_vax_date = start_date
  vax_count = 0
  data = c()
  us = 0
  overlap_vec = c()
  cache = c()
  for (i in 1:(length(ordered_event)-1)) {
    interval_start_date = ordered_date[i]
    interval_start_event = ordered_event[i]
    interval_end_date = ordered_date[i+1]
    interval_end_event = ordered_event[i+1]
    interval_length = month_diff(interval_end_date, interval_start_date)
    if (interval_length == 0) {
      cache = c(cache, substr(interval_start_event, 1, 1))
      next
    }
    # [rew, act, num_vax, months_to_last_vax, months_to_last_inf, inf]
    data_sub = matrix(0, nrow = interval_length, ncol = 6)
    overlap_vec_i = rep(NA, interval_length)
    
    # get reward, action and observations
    if (substr(interval_start_event, 1, 1) == "v") {
      data_sub[1, 2] = 1 # action
      data_sub[1, 1] = -0.5 # reward
      
      vax_count = vax_count + 1
      data_sub[, 3] = vax_count
      data_sub[, 4] = 0:(interval_length - 1)
      data_sub[, 5] = month_diff(interval_start_date, last_inf_date) + 0:(interval_length - 1)
      last_vax_date = interval_start_date
    } else if (substr(interval_start_event, 1, 1) == "i") {
      data_sub[, 3] = vax_count
      data_sub[, 4] = month_diff(interval_start_date, last_vax_date) + 0:(interval_length - 1)
      data_sub[, 5] = 0:(interval_length - 1)
      last_inf_date = interval_start_date
      
      data_sub[1, 6] = 1
    } else if (substr(interval_start_event, 1, 1) == "s") {
      data_sub[, 3] = 0
      data_sub[, 4] = -1 #month_diff(end_date, start_date)
      data_sub[, 5] = -1 #month_diff(end_date, start_date)
    } else {
      cat("undefined sequence")
      us = us + 1
      return(c())
    }
    
    if (substr(interval_end_event, 1, 1) == "i") {
      data_sub[nrow(data_sub), 1] = data_sub[nrow(data_sub), 1] - 10
    }
    
    if (last_vax_date == start_date) {
      data_sub[, 3] = 0
      data_sub[, 4] = -1 #month_diff(end_date, start_date)
    }
    
    if (last_inf_date == start_date) {
      data_sub[, 5] = -1 #month_diff(end_date, start_date)
    }
    
    cache = cache[cache %in% c("v", "i")]
    if (length(cache) > 0) {
      cache_num_vax = sum(cache == "v")
      cache_num_inf = sum(cache == "i")
      if (cache_num_vax > 0) {
        data_sub[1, 1] = data_sub[1, 1] - 0.5 * cache_num_vax
        data_sub[1, 2] = 1
        data_sub[, 3] = data_sub[, 3] + cache_num_vax
        data_sub[, 4] = 0:(interval_length - 1)
        last_vax_date = interval_start_date
        vax_count = vax_count + cache_num_vax
      }
      if (cache_num_inf > 0) {
        data[nrow(data), 1] = data[nrow(data), 1] - 10 * (cache_num_inf - (cache[1] == "i"))
        data_sub[, 5] = 0:(interval_length - 1)
        last_inf_date = interval_start_date
        
        data_sub[1, 6] = 1
      }
      overlap_vec_i[1] = paste(c(cache, substr(interval_start_event, 1, 1)), collapse = "")
    }
    cache = c()
    data = rbind(data, data_sub)
    overlap_vec = c(overlap_vec, overlap_vec_i)
  }
  date_idx = seq.Date(from = start_date, to = terminal_date, by = "month")
  date_idx = date_idx[1:(length(date_idx) - 1)]
  data = as.data.frame(data)
  data = cbind(rowdata$DEID_PatientID, date_idx, data, demo_info)
  if (terminal_date == end_date) {
    data = cbind(data, 
                 rep(0, nrow(data)), # terminal
                 c(rep(0, nrow(data)-1), 1)) # episode terminal
  } else {
    data = cbind(data, 
                 c(rep(0, nrow(data)-1), 1), # terminal
                 rep(0, nrow(data))) # episode terminal
  }
  data = cbind(data, overlap_vec)
  colnames(data) = c("id", "month", "reward", "action", "numVax", "monthsLastVax", "monthsLastInf", "inf",
                     demo_var, "terminal", "epis_terminal", "overlap")
  return(list(data, ordered_date, us))
}

create_long_data_month = function(data, n_batch, batch_size, end_date, start_date, demo_var) {
  n_months = month_diff(end_date, start_date)
  RLdata = c()
  total_num_patients = nrow(data)
  overlap = 0
  us = 0
  stop = FALSE
  for (batch in 1:n_batch) {
    data_batch = data[((batch-1)*batch_size+1):min((batch*batch_size), total_num_patients), ]
    num_rows = min(batch_size, total_num_patients - (batch-1)*batch_size)*n_months
    RLdata_batch = data.frame(matrix(0, nrow = num_rows, ncol = 9+length(demo_var)))
    RLdata_batch = cbind("id", rep(as.Date("2020-03-01"), num_rows), RLdata_batch)
    colnames(RLdata_batch) = c("id", "month", "reward", "action", "numVax", "monthsLastVax", "monthsLastInf", "inf",
                               demo_var, "terminal", "epis_terminal", "overlap")
    row_count = 0
    for (i in 1:batch_size) {
      rowdata = data_batch[i, ]
      d = create_long_data_month_i(rowdata, end_date, start_date, demo_var)
      if (length(d[[1]]) == 0) {next}
      RLdata_batch[(row_count+1):(row_count+nrow(d[[1]])), ] = d[[1]]
      row_count = row_count + nrow(d[[1]])
      us = us + d[[3]]
      #cat(i, row_count, "\n")
      if (((batch-1)*batch_size+i) >= total_num_patients) {
        stop = TRUE
        break
      }
    }
    RLdata_batch = RLdata_batch[RLdata_batch$id != "id", ]
    RLdata = rbind(RLdata, RLdata_batch)
    cat(paste(batch, "/", n_batch, sep = " "), "\n")
    if (stop) {break}
  }
  #overlap = sum(!is.na(RLdata$overlap))
  #cat(overlap, us)
  return(RLdata)
}