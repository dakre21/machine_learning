import coloredlogs, logging

# Create logger object for application
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')

# Global declarations
LOG_REG = "LogReg"
ARIMA   = "ARIMA"
GAM     = "GAM"

models = [
  LOG_REG,
  ARIMA,
  GAM
]

intervals = [
  "daily",
  "weekly",
  "monthly",
  "quarterly",
  "annual"
]
