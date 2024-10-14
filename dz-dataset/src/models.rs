use crate::schema::poems;

use diesel::prelude::*;

#[derive(Queryable, Selectable)]
#[diesel(table_name = poems)]
#[diesel(check_for_backend(diesel::pg::Pg))]
pub struct Poem {
    pub id: i32,
    pub author: Option<String>,
    pub title: String,
    pub topic: String,
    pub body: String,
}

#[derive(Insertable)]
#[diesel(table_name = poems)]
pub struct NewPoem {
    pub author: Option<String>,
    pub title: String,
    pub topic: String,
    pub body: String,
}
