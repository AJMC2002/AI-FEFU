-- Your SQL goes here
CREATE TABLE "poems"(
	"id" SERIAL NOT NULL PRIMARY KEY,
	"author" VARCHAR,
	"title" VARCHAR NOT NULL,
	"body" TEXT NOT NULL,
	"topic" VARCHAR NOT NULL
);

