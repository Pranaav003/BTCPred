"""Add model_artifacts table for DB-backed model storage.

Revision ID: 0002_model_artifacts
Revises: 0001_initial_schema
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_model_artifacts"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "model_artifacts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(64), unique=True, nullable=False),
        sa.Column("data", sa.LargeBinary(), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=True),
        sa.Column("model_type", sa.String(64), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("model_artifacts")
