mod lemmatization;
mod models;
mod schema;

use core::panic;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::io::Write;
use std::sync::LockResult;

use diesel::prelude::*;
use dotenvy::dotenv;
use lemmatization::*;
use models::*;
use polars::prelude::*;
use rayon::prelude::*;
use regex::Regex;
use schema::poems;
use tokenizers::Tokenizer;

fn establish_connection() -> PgConnection {
    dotenv().ok();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
}

#[tokio::main(core_threads = 16)]
async fn main() -> Result<(), Box<dyn Error + Send + Sync>> {
    env_logger::init();

    let conn = &mut establish_connection();
    let results: Vec<Poem> = poems::dsl::poems.select(Poem::as_select()).load(conn)?;

    let mut words: HashMap<String, i64> = HashMap::new();
    let re = Regex::new(r"\W+")?;

    for poem in results {
        for token in re.split(&poem.body) {
            let lowercase_token = token.to_lowercase();
            let count = words.get(&lowercase_token).unwrap_or(&0) + 1;
            words.insert(lowercase_token, count);
        }
    }

    println!("TOTAL NUMBER OF WORDS: {}", words.len());

    let mut df_schema = Schema::default();
    df_schema.insert("words".into(), DataType::String);
    df_schema.insert("frequency".into(), DataType::Int64);

    let mut df = DataFrame::from_rows_and_schema(
        words
            .par_iter()
            .map(|(k, &v)| polars::frame::row::Row::new(vec![k.as_str().into(), v.into()]))
            .collect::<Vec<_>>()
            .as_slice(),
        &df_schema,
    )?;
    println!("ORIGINAL DF\n{:?}\n", df);

    let most_frequent_df = df
        .sort(
            ["frequency"],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_multithreaded(true),
        )?
        .head(Some(15));
    println!("MOST FREQUENT DF\n{:?}\n", most_frequent_df);

    let least_frequent_df = df
        .sort(
            ["frequency"],
            SortMultipleOptions::default().with_multithreaded(true),
        )?
        .head(Some(15));
    println!("LEAST FREQUENT DF\n{:?}\n", least_frequent_df);

    let head_pct = 0 as f64 / df.height() as f64; // Percent removed from the most popular words
    let tail_pct = 0.9; // Percent removed from the least popular words
    let offset = (df.height() as f64 * head_pct).round() as i64;
    let length = (df.height() as f64 * (1.0 - (head_pct + tail_pct))).round() as usize;
    df.sort_in_place(
        ["frequency"],
        SortMultipleOptions::default()
            .with_order_descending(true)
            .with_multithreaded(true),
    )?;
    df = df.slice(offset.into(), length.into());
    println!("CLEAN DF\n{:?}\n", df);

    let mut fout = std::fs::File::create("custom.tokens")?;
    for s in df["words"].str()?.iter() {
        fout.write(format!("{}\n", s.unwrap()).as_bytes())?;
    }

    let tokenizer = Tokenizer::from_pretrained("bert-base-cased", None)?;
    let tokens_series: Vec<Series> = df["words"]
        .str()?
        .par_iter()
        .map(|word| {
            tokenizer
                .encode(word.unwrap(), false)
                .unwrap()
                .get_tokens()
                .iter()
                .map(|t| t.as_str())
                .collect()
        })
        .collect();
    df.insert_column(1, Series::new("tokens".into(), tokens_series))?;
    println!("TOKENIZED DF\n{:?}\n", df);

    let lemma_map = read_map("lemmas.csv")?;
    let lemma_series = df["words"]
        .str()?
        .par_iter()
        .map(|word| Ok(lemmatize(word.unwrap(), &lemma_map)?[0].to_owned()))
        .collect::<LockResult<Vec<String>>>()?;
    df.insert_column(2, Series::new("lemmas".into(), lemma_series))?;
    println!("LEMMATIZED DF\n{:?}\n", df);

    let lemma_iter = df.column("lemmas")?.str()?.into_iter();
    let count_iter = df.column("frequency")?.i64()?.into_iter();

    let mut tokens: HashMap<String, i64> = HashMap::new();

    for (lemma, count) in lemma_iter.zip(count_iter) {
        match (lemma, count) {
            (Some(lemma), Some(count)) => {
                let upd_count = tokens.get(lemma).unwrap_or(&0) + count;
                tokens.insert(lemma.to_owned(), upd_count);
            }
            (_, _) => (),
        }
    }

    let mut df_schema = Schema::default();
    df_schema.insert("tokens".into(), DataType::String);
    df_schema.insert("frequency".into(), DataType::Int64);

    let mut token_df = DataFrame::from_rows_and_schema(
        tokens
            .par_iter()
            .map(|(k, &v)| polars::frame::row::Row::new(vec![k.as_str().into(), v.into()]))
            .collect::<Vec<_>>()
            .as_slice(),
        &df_schema,
    )?;
    token_df.sort_in_place(
        ["frequency"],
        SortMultipleOptions::default()
            .with_order_descending(true)
            .with_multithreaded(true),
    )?;

    println!("TOKEN DF\n{:?}", token_df);

    Ok(())
}
