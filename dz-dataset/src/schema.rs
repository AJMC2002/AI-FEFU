// @generated automatically by Diesel CLI.

diesel::table! {
    poems (id) {
        id -> Int4,
        author -> Nullable<Varchar>,
        title -> Varchar,
        body -> Text,
        topic -> Varchar,
    }
}
