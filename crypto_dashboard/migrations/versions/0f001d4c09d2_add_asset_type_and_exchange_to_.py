"""Add asset_type and exchange to Portfolio and Trade models (again)

Revision ID: 0f001d4c09d2
Revises: b58d67d491c5
Create Date: 2025-06-07 21:33:24.534520

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '0f001d4c09d2'
down_revision = 'b58d67d491c5'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands adjusted by developer for data preservation ###
    with op.batch_alter_table('portfolio', schema=None) as batch_op:
        batch_op.alter_column('crypto_id', new_column_name='asset_symbol', existing_type=sa.String(length=50), nullable=False)
        batch_op.add_column(sa.Column('asset_type', sa.String(length=20), nullable=False, server_default='crypto'))
        batch_op.add_column(sa.Column('exchange', sa.String(length=50), nullable=True))

    with op.batch_alter_table('trade', schema=None) as batch_op:
        batch_op.alter_column('crypto_id', new_column_name='asset_symbol', existing_type=sa.String(length=50), nullable=False)
        batch_op.add_column(sa.Column('asset_type', sa.String(length=20), nullable=False, server_default='crypto'))
        batch_op.add_column(sa.Column('exchange', sa.String(length=50), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands adjusted by developer for data preservation ###
    with op.batch_alter_table('trade', schema=None) as batch_op:
        batch_op.alter_column('asset_symbol', new_column_name='crypto_id', existing_type=sa.String(length=50), nullable=False)
        batch_op.drop_column('exchange')
        batch_op.drop_column('asset_type')

    with op.batch_alter_table('portfolio', schema=None) as batch_op:
        batch_op.alter_column('asset_symbol', new_column_name='crypto_id', existing_type=sa.String(length=50), nullable=False)
        batch_op.drop_column('exchange')
        batch_op.drop_column('asset_type')
    # ### end Alembic commands ###
