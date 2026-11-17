from alembic import context
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url
from sqlalchemy.pool import NullPool

from app.config import Settings

database_url = make_url(Settings().database_url).set(drivername="postgresql+psycopg")
version_table_schema = context.config.attributes.get("version_table_schema")

if context.is_offline_mode():
    context.configure(
        url=database_url, literal_binds=True, version_table_schema=version_table_schema
    )
    with context.begin_transaction():
        context.run_migrations()
else:
    engine = create_engine(database_url, poolclass=NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, version_table_schema=version_table_schema)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()
