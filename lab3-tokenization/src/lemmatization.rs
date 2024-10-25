use std::{
    collections::HashMap,
    sync::{Arc, LockResult, Mutex},
};

use rayon::prelude::*;

// All functions taken and improved are from rust_lemmatizer

pub fn read_map(path: &str) -> LockResult<HashMap<String, String>> {
    let lemma_map = Arc::new(Mutex::new(HashMap::<String, String>::new()));
    let mut rdr = csv::Reader::from_path(path).unwrap();

    rdr.records().par_bridge().for_each(|rec| {
        let rr = rec.unwrap();
        let lemma = rr.get(0).unwrap();
        let derivatives = rr.get(1).unwrap();
        lemma_map
            .lock()
            .unwrap()
            .insert(lemma.into(), derivatives.into());
    });

    Arc::try_unwrap(lemma_map).unwrap().into_inner()
}

pub fn lemmatize(
    string_to_analyze: &str,
    lemma_map: &HashMap<String, String>,
) -> LockResult<Vec<String>> {
    let mut word_list: Vec<String> = Vec::new();
    for split_word in string_to_analyze.split(&[' ', '\''][..]) {
        word_list.push(split_word.to_string());
    }

    let lemma_string = Arc::new(Mutex::new(Vec::new()));

    word_list.par_iter().for_each(|line| {
        let mut lemma_line: Vec<String> = Vec::new();
        let mut changed_words: Vec<String> = Vec::new();
        for word in line.split_whitespace() {
            for (lemma, derivatives) in lemma_map {
                if derivatives.contains(",") {
                    let split = derivatives.split(",");
                    for s in split {
                        if s.trim() == word {
                            lemma_line.push(lemma.to_string());
                            changed_words.push(word.to_string());
                        }
                    }
                } else if derivatives == word {
                    lemma_line.push(lemma.to_string());
                    changed_words.push(word.to_string());
                }
            }
            if changed_words.contains(&word.to_string()) {
            } else {
                lemma_line.push(word.to_string());
            }
        }
        lemma_string.lock().unwrap().push(lemma_line.join(" "));
    });

    Arc::try_unwrap(lemma_string).unwrap().into_inner()
}
