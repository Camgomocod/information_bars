#!/bin/bash
# ================================================================================
# backup_db.sh — Backup automático de TimescaleDB
# ================================================================================
# Uso manual:
#   ./scripts/backup_db.sh
#
# Uso en cron (semanal):
#   0 2 * * 0 cd /path/to/trading-core && ./scripts/backup_db.sh
#
# Restore:
#   pg_restore -d trading_core -U trading backups/trading_core_YYYYMMDD.dump
# ================================================================================

set -euo pipefail

BACKUP_DIR="backups"
DB_NAME="trading_core"
DB_USER="trading"
TIMESTAMP=$(date +%Y%m%d_%H%M)
FILENAME="trading_core_${TIMESTAMP}.dump"
KEEP_DAYS=30

# Crear directorio de backups si no existe
mkdir -p "${BACKUP_DIR}"

echo "========================================"
echo "  BACKUP TIMESCALEDB"
echo "  Fecha: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"

# Verificar que el contenedor está corriendo
if ! docker ps | grep -q "trading-core-timescaledb"; then
    echo "⚠️  Contenedor TimescaleDB no está corriendo. Abortando."
    exit 1
fi

# Realizar backup
echo "📦 Exportando base de datos..."
docker exec -t trading-core-timescaledb pg_dump -Fc -U "${DB_USER}" -d "${DB_NAME}" > "${BACKUP_DIR}/${FILENAME}"

# Verificar que el backup se creó correctamente
if [ -f "${BACKUP_DIR}/${FILENAME}" ] && [ -s "${BACKUP_DIR}/${FILENAME}" ]; then
    SIZE=$(du -h "${BACKUP_DIR}/${FILENAME}" | cut -f1)
    echo "✅ Backup completado: ${FILENAME} (${SIZE})"
else
    echo "❌ Error: backup vacío o no se creó"
    exit 1
fi

# Limpiar backups antiguos
echo "🧹 Limpiando backups antiguos (>${KEEP_DAYS} días)..."
find "${BACKUP_DIR}" -name "trading_core_*.dump" -mtime +${KEEP_DAYS} -delete

echo ""
echo "📊 Backups disponibles:"
ls -lh "${BACKUP_DIR}"/trading_core_*.dump 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'

echo ""
echo "========================================"
echo "  BACKUP FINALIZADO"
echo "========================================"
