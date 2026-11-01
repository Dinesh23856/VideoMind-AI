"""add_vector_embeddings_table

Revision ID: a1b2c3d4e5f6
Revises: 4e3ae7c65c5f
Create Date: 2026-10-04 11:15:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '4e3ae7c65c5f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema to include vector_embeddings table."""
    conn = op.get_bind()
    inspector = sa.inspect(conn)
    if 'vector_embeddings' not in inspector.get_table_names():
        op.create_table(
            'vector_embeddings',
            sa.Column('id', sa.String(length=64), nullable=False),
            sa.Column('document_id', sa.String(length=128), nullable=False),
            sa.Column('tenant_id', sa.String(length=64), nullable=False, server_default='default'),
            sa.Column('workspace_id', sa.String(length=64), nullable=False, server_default='default'),
            sa.Column('user_id', sa.String(length=64), nullable=True),
            sa.Column('embedding_model', sa.String(length=128), nullable=False),
            sa.Column('embedding_version', sa.String(length=32), nullable=False, server_default='1.0.0'),
            sa.Column('embedding_dimension', sa.Integer(), nullable=False),
            sa.Column('modality', sa.String(length=32), nullable=False, server_default='transcript'),
            sa.Column('source_id', sa.String(length=128), nullable=True),
            sa.Column('start_ms', sa.Integer(), nullable=True),
            sa.Column('end_ms', sa.Integer(), nullable=True),
            sa.Column('text', sa.Text(), nullable=False),
            sa.Column('vector_json', sa.Text(), nullable=False),
            sa.Column('metadata_json', sa.Text(), nullable=False, server_default='{}'),
            sa.Column('created_at', sa.Float(), nullable=False),
            sa.PrimaryKeyConstraint('id')
        )
        op.create_index(op.f('ix_vector_embeddings_id'), 'vector_embeddings', ['id'], unique=False)
        op.create_index(op.f('ix_vector_embeddings_document_id'), 'vector_embeddings', ['document_id'], unique=False)
        op.create_index(op.f('ix_vector_embeddings_tenant_id'), 'vector_embeddings', ['tenant_id'], unique=False)
        op.create_index(op.f('ix_vector_embeddings_workspace_id'), 'vector_embeddings', ['workspace_id'], unique=False)
        op.create_index(op.f('ix_vector_embeddings_user_id'), 'vector_embeddings', ['user_id'], unique=False)
        op.create_index('idx_vector_tenant_workspace', 'vector_embeddings', ['tenant_id', 'workspace_id'], unique=False)
        op.create_index('idx_vector_model_dim', 'vector_embeddings', ['embedding_model', 'embedding_dimension'], unique=False)


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('idx_vector_model_dim', table_name='vector_embeddings')
    op.drop_index('idx_vector_tenant_workspace', table_name='vector_embeddings')
    op.drop_index(op.f('ix_vector_embeddings_user_id'), table_name='vector_embeddings')
    op.drop_index(op.f('ix_vector_embeddings_workspace_id'), table_name='vector_embeddings')
    op.drop_index(op.f('ix_vector_embeddings_tenant_id'), table_name='vector_embeddings')
    op.drop_index(op.f('ix_vector_embeddings_document_id'), table_name='vector_embeddings')
    op.drop_index(op.f('ix_vector_embeddings_id'), table_name='vector_embeddings')
    op.drop_table('vector_embeddings')
