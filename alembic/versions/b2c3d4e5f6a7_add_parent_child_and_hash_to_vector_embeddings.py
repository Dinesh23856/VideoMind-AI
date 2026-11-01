"""add_parent_child_and_hash_to_vector_embeddings

Revision ID: b2c3d4e5f6a7
Revises: a1b2c3d4e5f6
Create Date: 2026-10-04 14:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Safely add parent_id, parent_content, content_hash, and updated_at to vector_embeddings."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    tables = inspector.get_table_names()
    if 'vector_embeddings' not in tables:
        return
    existing_cols = {c['name'] for c in inspector.get_columns('vector_embeddings')}
    existing_indexes = {i['name'] for i in inspector.get_indexes('vector_embeddings')}

    with op.batch_alter_table('vector_embeddings') as batch_op:
        if 'parent_id' not in existing_cols:
            batch_op.add_column(sa.Column('parent_id', sa.String(length=128), nullable=True))
        if 'parent_content' not in existing_cols:
            batch_op.add_column(sa.Column('parent_content', sa.Text(), nullable=True))
        if 'content_hash' not in existing_cols:
            batch_op.add_column(sa.Column('content_hash', sa.String(length=64), nullable=True))
        if 'updated_at' not in existing_cols:
            batch_op.add_column(sa.Column('updated_at', sa.Float(), nullable=True))
        if 'idx_vector_parent' not in existing_indexes:
            batch_op.create_index('idx_vector_parent', ['parent_id'], unique=False)
        if 'idx_vector_hash' not in existing_indexes:
            batch_op.create_index('idx_vector_hash', ['content_hash'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    with op.batch_alter_table('vector_embeddings') as batch_op:
        batch_op.drop_index('idx_vector_hash')
        batch_op.drop_index('idx_vector_parent')
        batch_op.drop_column('updated_at')
        batch_op.drop_column('content_hash')
        batch_op.drop_column('parent_content')
        batch_op.drop_column('parent_id')
