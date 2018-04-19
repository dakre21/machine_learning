import coloredlogs, logging

# Create logger object for application
logger = logging.getLogger(__name__)

coloredlogs.install(level='DEBUG')

# Global declarations
# Models
LIN_REG   = "LinearRegression"
BAGGING   = "Bagging"
#SGD       = "SGD"
#PAL       = "PAL"
LASSO     = "LASSO"

# Keys
SYM_ONE    = 'SYM_ONE'
SYM_TWO    = 'SYM_TWO'
API_KEY    = 'API_KEY'
LABEL      = '_LABEL'
START_DATE = "START_DATE"

models = [
  LIN_REG,
  BAGGING,
  SGD,
  PAL,
  LASSO
]

intervals = [
  "daily",
  "weekly",
  "monthly",
  "quarterly",
  "annual"
]
