#
# Add cluster specific environment variables in this file.
#

#
# Set Java options, i.e., -Dproperty=xyz
#
#JAVA_OPTS=

#
# To use Hibernate backed MapStorePkDbImpl, set the following property and
# configure MapStorePkDbImpl in the $CLUSTER_DIR/etc/hazelcast.xml file.
# MySQL and PostgreSQL Hibernate configuration files are provided to get
# you started. You should copy one of them and enter your DB information.
# You can include your JDBC driver in the ../pom.xml file and run ./build_app
# which downloads and places it in the $PADOGRID_WORKSPACE/lib
# directory. CLASSPATH includes all the jar files in that directory for
# the apps and clusters running in this workspace.
#
#JAVA_OPTS="$JAVA_OPTS -Dhazelcast-addon.hibernate.config=$CLUSTER_DIR/etc/hibernate.cfg-mysql.xml"
#JAVA_OPTS="$JAVA_OPTS -Dhazelcast-addon.hibernate.config=$CLUSTER_DIR/etc/hibernate.cfg-postgresql.xml"
#JAVA_OPTS="$JAVA_OPTS -Dhazelcast-addon.hibernate.config=$CLUSTER_DIR/etc/hibernate.cfg-derby.xml"

#
# Set Management Center Java options, i.e., -Dhazelcast.mc.forceLogoutOnMultipleLogin=true
#
#MC_JAVA_OPTS=

#
# Set RUN_SCRIPT. Absolute path required.
# If set, the 'start_member' command will run this script instead of com.hazelcast.core.server.StartServer.
# Your run script will inherit the following:
#    JAVA      - Java executable.
#    JAVA_OPTS - Java options set by padogrid.
#    CLASSPATH - Class path set by padogrid. You can include additional libary paths.
#                You should, however, place your library files in the plugins directories
#                if possible.
#    CONFIG_FILE - Config file path. You can override the config file path.
#
# Run Script Example:
#    "$JAVA" $JAVA_OPTS com.newco.MyMember &
#
# Although it is not required, your script should be placed in the bin_sh directory.
#
#RUN_SCRIPT=$CLUSTER_DIR/bin_sh/your-script

CONFIG_FILE=$CLUSTER_DIR/etc/hazelcast.yaml

JAVA_OPTS="$JAVA_OPT -Dhazelcast.member.naming.moby.enabled=false"
export PYTHONPATH=$PADOGRID_WORKSPACE/apps/ml_lstm/src/main/python
