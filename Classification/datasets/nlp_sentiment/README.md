# Sentiment Labelled Sentences

Unmodified UTF-8 source files from the UCI Machine Learning Repository, dataset
331, downloaded on 2026-10-16. Each nonempty record has `sentence<TAB>label`,
where 0 means negative sentiment and 1 means positive sentiment. There are
1,000 labelled sentences per source: Amazon, IMDb and Yelp. Two IMDb records
wrap across physical lines. The loader accumulates text until the terminal
tab + binary label, rather than discarding the continuation fragments.
Quoted text is treated literally; these files are not parsed as quoted CSV.

- Creator: Dimitrios Kotzias (2015).
- Dataset: https://doi.org/10.24432/C57604
- Source: https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Original archive: https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip
- Archive SHA-256: `afc26626d710899948693e1a61405dce197f57ffa719fa1130d346b4cc095343`.

Associated paper: Kotzias, D., Denil, M., de Freitas, N., and Smyth, P. (2015),
*From Group to Individual Labels Using Deep Features*, KDD.

The experiment normalizes whitespace and Unicode for duplicate detection,
removes conflicting duplicate labels, and keeps one record per remaining
normalized text before splitting. These transformations happen in memory;
the downloaded files are unchanged. This is a review-sentiment example, with
no demographic inference target.
