import sqlalchemy
import sqlite3


def make_db():
    aff_query = '''
    CREATE TABLE IF NOT EXISTS affiliations (
        id INTEGER PRIMARY KEY,
        name VARCHAR(250) NOT NULL,
        zip INT
    );'''

    author_query = '''
    CREATE TABLE IF NOT EXISTS authors (
        first_name VARCHAR(250),
        last_name VARCHAR(250) NOT NULL,
        affiliation_id INTEGER REFERENCES affiliations(id)
    );'''

    sqliteConnection = sqlite3.connect('test_sqlite.db')
    cursor = sqliteConnection.cursor()
    print("Database created and Successfully Connected to SQLite")

    cursor.execute(aff_query)
    cursor.execute(author_query)

    cursor.close()


def append_tO_db(df: pd.DataFrame):
    # Do some operations to split affilaitions and authors

    engine = sqlalchemy.create_engine(
        'sqlite:////Users/arthur/teaching/duq_ds3_2023/data/live_test_sqlite.db')
    df.to_sql('aff', engine, index=False, if_exists='append')

    # Have to account for SQLAlchemy v2
    # with engine.connect() as conn:
    #     df2 = pd.read_sql(sqlalchemy.text('SELECT * FROM aff'), conn)
