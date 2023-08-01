#
# Enter app specifics in this file.
#

# Cluster level variables:
# ------------------------
# BASE_DIR - hazelcast-addon base dir
# ETC_DIR - Cluster etc dir
# LOG_DIR - Cluster log dir

# App level variables:
# --------------------
# APPS_DIR - <hazelcast-addon>/apps dir
# APP_DIR - App base dir
# APP_ETC_DIR - App etc dir
# APP_LOG_DIR - App log dir

# Set JAVA_OPT to include your app specifics.
#JAVA_OPTS=$JAVA_OPTS

# Pado version - set by build_app. This value may change if
# build_app determines a different Pado version during the build.
PADO_VERSION=0.5.0-B1-SNAPSHOT

# Pado Home
PADO_HOME=$APP_DIR/pado_${PADO_VERSION}

# Set app specific class path.
#CLASSPATH="$CLASSPATH"
CLASSPATH="${PADO_HOME}/plugins/*:${PADO_HOME}/lib/*"

# Python path to the ML code
export PYTHONPATH="$APP_DIR/src/main/python"
