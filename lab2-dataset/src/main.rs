mod models;
mod schema;

use core::panic;
use std::collections::HashMap;
use std::error::Error;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;
use std::{env, fs, io};

use diesel::{insert_into, prelude::*};
use dotenvy::dotenv;
use glob::glob;
use itertools::Itertools;
use kaggle::KaggleApiClient;
use log::info;
use models::*;
use polars::datatypes::DataType;
use polars::frame::row::Row;
use polars::frame::DataFrame;
use polars::prelude::{Schema, SortMultipleOptions};
use regex::Regex;
use schema::poems;

fn establish_connection() -> PgConnection {
    dotenv().ok();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
}

fn redo_table() -> Result<(), Box<dyn Error>> {
    let output = Command::new("diesel")
        .arg("migration")
        .arg("redo")
        .output()
        .expect("Error when resetting the table.");
    info!("Redo status: {}", output.status);
    io::stdout().write_all(&output.stdout).unwrap();
    io::stderr().write_all(&output.stderr).unwrap();
    Ok(())
}

async fn download_dataset(dataset: &str) -> Result<(), Box<dyn Error>> {
    let kaggle = KaggleApiClient::builder().build()?;
    let res = kaggle
        .dataset_download_all_files(dataset, None, None)
        .await?;

    info!("Zip downloaded in {}.", res.display());

    if cfg!(target_os = "windows") {
        panic!("Utility `unzip` not available. Expected UNIX based OS.");
    } else {
        Command::new("unzip")
            .arg(res)
            .arg("-d")
            .arg("data")
            .output()
            .expect("Unzip failed.");

        info!("Unzipped dataset.");
    };

    Ok(())
}

fn path_to_poem(path: &PathBuf) -> NewPoem {
    let re = Regex::new(r"^(?<topic>.+?)Poems(?<title>.+)Poemby(?<author>.*)$").unwrap();
    let filename = path.file_stem().unwrap();

    info!("Reading file: {}.txt", &filename.to_str().unwrap());

    let caps = re.captures(filename.to_str().unwrap()).unwrap();

    let topic = caps["topic"].to_string().to_lowercase();
    let title = caps["title"].to_string();
    let author = if &caps["author"] == "" {
        None
    } else {
        Some(caps["author"].to_string())
    };
    let body = fs::read_to_string(path).expect("Couldn't read path.");

    NewPoem {
        author,
        title,
        topic,
        body,
    }
}

fn insert_poems(conn: &mut PgConnection) -> Result<(), Box<dyn Error>> {
    let paths = glob("./data/topics/**/*.txt")?;
    let mut n_rows = 0;
    paths.chunks(100).into_iter().for_each(|chunk| {
        let new_poem_chunk = chunk
            .map(|entry| match entry {
                Ok(path) => path_to_poem(&path),
                Err(e) => panic!("Error with path: {:?}", e),
            })
            .collect::<Vec<_>>();

        insert_into(poems::table)
            .values(&new_poem_chunk)
            .execute(conn)
            .expect("Couldn't insert values into the database.");

        n_rows += new_poem_chunk.len();
    });
    info!("Inserted {} rows.", n_rows);

    Ok(())
}

#[tokio::main(core_threads = 16)]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    if env::var("REDO").is_ok() {
        redo_table()?;
        download_dataset("michaelarman/poemsdataset").await?;
        let conn = &mut establish_connection();
        insert_poems(conn)?;
    }

    let conn = &mut establish_connection();
    let results: Vec<Poem> = poems::dsl::poems.select(Poem::as_select()).load(conn)?;

    let mut tokens: HashMap<String, i64> = HashMap::new();
    let re = Regex::new(r"\W+")?;

    for poem in results {
        for token in re.split(&poem.body) {
            let lowercase_token = token.to_lowercase();
            let count = tokens.get(&lowercase_token).unwrap_or(&0) + 1;
            tokens.insert(lowercase_token, count);
        }
    }

    let mut df_schema = Schema::default();
    df_schema.insert("token".into(), DataType::String);
    df_schema.insert("frequency".into(), DataType::Int64);

    let df = DataFrame::from_rows_and_schema(
        tokens
            .iter()
            .map(|(k, &v)| Row::new(vec![k.as_str().into(), v.into()]))
            .collect::<Vec<Row>>()
            .as_slice(),
        &df_schema,
    )?;

    let most_frequent_df = df
        .sort(
            ["frequency"],
            SortMultipleOptions::default()
                .with_order_descending(true)
                .with_multithreaded(true),
        )?
        .head(Some(10));
    println!("{:?}", most_frequent_df);

    let least_frequent_df = df
        .sort(
            ["frequency"],
            SortMultipleOptions::default().with_multithreaded(true),
        )?
        .head(Some(10));
    println!("{:?}", least_frequent_df);

    Ok(())
}
