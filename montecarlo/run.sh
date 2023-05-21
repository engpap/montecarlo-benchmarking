# Give execute permissions to the script using: chmod +x /home/ubuntu/montecarlo-benchmarking/montecarlo/run.sh

# Navigate to the project folder
cd /home/ubuntu/montecarlo-benchmarking/montecarlo

# Compile
#mvn -X -e package 
mvn package 

# Run with GraalVM
/home/ubuntu/graalvm-ce-java11-22.1.0/bin/java -cp target/montecarlo-1.0-SNAPSHOT.jar com.polimi.montecarlo.App
#/home/ubuntu/graalvm-ce-java11-22.1.0/bin/java -cp target/montecarlo-1.0-SNAPSHOT.jar com.polimi.montecarlo.B1

#mvn -Dtest=TestBenchmark test