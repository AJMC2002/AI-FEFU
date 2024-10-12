mod models;
mod schema;

use core::panic;
use std::error::Error;
use std::path::PathBuf;
use std::process::Command;
use std::{env, fs};

use models::*;
use schema::poems;

use diesel::{insert_into, prelude::*};
use dotenvy::dotenv;
use glob::glob;
use itertools::Itertools;
use kaggle::KaggleApiClient;
use log::{debug, info};
use regex::Regex;

fn establish_connection() -> PgConnection {
    dotenv().ok();

    let database_url = env::var("DATABASE_URL").expect("DATABASE_URL must be set");
    PgConnection::establish(&database_url)
        .unwrap_or_else(|_| panic!("Error connecting to {}", database_url))
}

fn redo_table() -> Result<(), Box<dyn Error>> {
    Command::new("diesel")
        .arg("migaration")
        .arg("redo")
        .output()
        .expect("Error when resetting the table.");
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
    let re = Regex::new(r"^(?<topic>.+)Poems(?<title>.+)Poemby(?<author>.*)$").unwrap();
    let filename = path.file_stem().unwrap();

    debug!("Reading file: {}.txt", &filename.to_str().unwrap());

    let caps = re.captures(filename.to_str().unwrap()).unwrap();

    let topic = caps["topic"].to_string();
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

#[tokio::main(core_threads = 8)]
async fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();

    if env::var("REDO").is_ok() {
        redo_table()?;
        download_dataset("michaelarman/poemsdataset").await?;
        let conn = &mut establish_connection();
        insert_poems(conn)?;
    }

    let conn = &mut establish_connection();
    let results = poems::dsl::poems
        .select(Poem::as_select())
        .load(conn)
        .expect("Rizz");

    Ok(())
}
