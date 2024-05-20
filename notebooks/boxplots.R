library(jsonlite)

# Set encoding
encodings <- c("UTF-8")

# Read JSON output
json_info<-jsonlite::fromJSON('data/output/model_10h/output_test_model_10h.json', simplifyDataFrame = FALSE)

# Set list of phones (vocab)
phones <- c('<pad>', '<s>', '</s>', '<unk>', '|', 'a', 'b', 'd', 'dʒ', 'e', 'f', 
              'i', 'j', 'j̃', 'k', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't', 'tʃ', 
              'u', 'v', 'w', 'w̃', 'z', 'ã', 'õ', 'ĩ', 'ũ', 'ɔ', 'ɛ', 'ɡ', 'ɲ', 
              'ʃ', 'ʎ', 'ʒ', 'χ', 'ẽ')

phones_dict <- list()

# Add phones probs
add_to_dictionary <- function(dictionary, key, value) {
  if (is.null(dictionary[[key]])) {
    dictionary[[key]] <- c(value)
  } else {
    dictionary[[key]] <- c(dictionary[[key]], value)
  }
  return(dictionary)
}

average_probs<-numeric()

# Iterate through audios
for (i in 1:length(json_info)){
  audio <- json_info[[i]]

  # Iterate through words
  for (j in 1:length(audio$words)){
    words <- audio$words[[j]]
    num_tokens <- length(words$tokens)

    if (num_tokens > 0){
      # Iterate through tokens and fetch their probs
      for (k in 1: num_tokens){
          token <- words$tokens[[k]]
          index <- which(phones == token$label)
          phones_dict <- add_to_dictionary(phones_dict, phones[index], token$score)
      }
    }
  }
}

phones_dict<-lapply(phones_dict, function(x) as.numeric(x))

average_probs<-sapply(phones_dict, function(x) mean(x))

values <- lapply(names(phones_dict), function(key) phones_dict[[key]])

phones_counts<-numeric()

# Add phones counts for the boxplot visualization
for(i in seq_along(values)){
  phones_counts[i]<-length(values[[i]])
}

par(family ="Ubuntu Mono")

# Plot info
boxplot(
  values,
  xlab = "Phoneme",  
  ylab = "Score",
  names = names(phones_dict)
)

# Add text labels with phone counts
text(x = 1:length(names(phones_dict)), y = par("usr")[4] + 0.1, labels = phones_counts, xpd = TRUE, srt = 90, adj = 1)

# Print phones
print(names(phones_dict))