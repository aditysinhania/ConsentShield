"""Generic Alembic revision template."""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # Initial schema is created via Base.metadata.create_all for local bootstrap.
    # Replace with explicit op.create_table migrations before production.
    pass


def downgrade() -> None:
    pass
