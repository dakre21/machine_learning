import coloredlogs, logging

# Create logger object for application
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')

# Global declarations
LIN_REG = "LinReg"
ARIMA   = "ARIMA"
GAM     = "GAM"

models = [
  LIN_REG,
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
