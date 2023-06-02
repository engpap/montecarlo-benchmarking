package com.polimi.montecarlo;

import static org.junit.Assert.assertEquals;

import java.io.IOException;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import org.json.JSONArray;
import org.json.JSONObject;


public class TestRunner {
    public static void main(String[] args) throws IOException, InterruptedException {
        int[] num_elems = {1024};
        int[] num_blocks = {1024};
        String[] choose_device_policies = {"round-robin"};
        String[] scaling_choices = {"strong"};
        boolean[] prefecth_values = {true};
        int[] num_gpus = {2};
        int[] block_sizes1d = {256};

        int i;
        for(int num_elem: num_elems){
            for(int num_block: num_blocks){
                for(String choose_device_policy: choose_device_policies){
                    for(String scaling_choice: scaling_choices){
                        for(boolean prefecth_val: prefecth_values){
                            for(int num_gpu: num_gpus){
                                for(int block_size1d: block_sizes1d){
                                    for(i=0; i<7; i++){
                                        create_json_config_file(false, num_elem, num_block, choose_device_policy, scaling_choice, prefecth_val, num_gpu, block_size1d);
                                        run_mvn_test();
                                    }
                                    run_nvprof_mvn_test();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private static void create_json_config_file(boolean nvprof_profile,int num_elem, int numBlock, String choose_device_policy, String scaling_choice, boolean prefecth_val, int num_gpus, int block_size1d) throws IOException {
        String PATH_TO_CONFIG_FILE = "/home/ubuntu/montecarlo-benchmarking/montecarlo/src/test/java/com/polimi/montecarlo/config_V100.json";
        FileReader fileReader = new FileReader(PATH_TO_CONFIG_FILE);
        StringBuilder jsonStrBuilder = new StringBuilder();
        int i;
        while ((i = fileReader.read()) != -1) {
            jsonStrBuilder.append((char) i);
        }
        fileReader.close();

        String jsonString = jsonStrBuilder.toString();
        JSONObject jsonObject = new JSONObject(jsonString);


        //  "Device_Selection_Policy,Prefetch,Problem_scaling,Number_of_GPUs,Total_number_of_options,Number_of_paths,Init_time_ms,Execution_time_ms";
        jsonObject.put("num_iter", 1);
        jsonObject.put("reAlloc", true);
        jsonObject.put("reInit", true);
        jsonObject.put("randomInit", false);
        jsonObject.put("cpuValidation", true);
        jsonObject.put("heap_size", 470);
        jsonObject.put("debug", true);
        jsonObject.put("nvprof_profile", nvprof_profile);

        JSONObject numElem = jsonObject.getJSONObject("num_elem");
        JSONArray monteCarloNumElem = numElem.getJSONArray("MonteCarlo");
        monteCarloNumElem.put(0, num_elem);

        JSONArray benchmarks = jsonObject.getJSONArray("benchmarks");
        benchmarks.put(0, "MonteCarlo");

        JSONObject numBlocks = jsonObject.getJSONObject("numBlocks");
        numBlocks.put("MonteCarlo", numBlock);

        JSONArray execPolicies = jsonObject.getJSONArray("exec_policies");
        execPolicies.put(0, "async");

        JSONArray dependencyPolicies = jsonObject.getJSONArray("dependency_policies");
        dependencyPolicies.put(0, "with-const");

        JSONArray newStreamPolicies = jsonObject.getJSONArray("new_stream_policies");
        newStreamPolicies.put(0, "always-new");

        JSONArray parentStreamPolicies = jsonObject.getJSONArray("parent_stream_policies");
        parentStreamPolicies.put(0, "disjoint");

        JSONArray chooseDevicePolicies = jsonObject.getJSONArray("choose_device_policies");
        chooseDevicePolicies.put(0, choose_device_policy);

        JSONArray memoryAdvise = jsonObject.getJSONArray("memory_advise");
        memoryAdvise.put(0, "none");

        JSONArray scalingChoice = jsonObject.getJSONArray("scalingChoice");
        scalingChoice.put(0, scaling_choice);

        JSONArray prefetch = jsonObject.getJSONArray("prefetch");
        prefetch.put(0, prefecth_val);

        JSONArray streamAttach = jsonObject.getJSONArray("stream_attach");
        streamAttach.put(0, false);

        JSONArray timeComputation = jsonObject.getJSONArray("time_computation");
        timeComputation.put(0, false);

        JSONArray numGPUs = jsonObject.getJSONArray("num_gpus");
        numGPUs.put(0, num_gpus);

        JSONObject blockSize1d = jsonObject.getJSONObject("block_size1d");
        blockSize1d.put("MonteCarlo", block_size1d);

        JSONObject blockSize2d = jsonObject.getJSONObject("block_size2d");
        blockSize2d.put("B2", 16);

        // Write the modified JSON back to the file
        FileWriter fileWriter = new FileWriter(PATH_TO_CONFIG_FILE);
        fileWriter.write(jsonObject.toString());
        fileWriter.flush();
        fileWriter.close();

    }

    private static void run_mvn_test() throws IOException, InterruptedException {
        ProcessBuilder builder = new ProcessBuilder();
        builder.command("./run.sh");
        builder.directory(new File("/home/ubuntu/montecarlo-benchmarking/montecarlo/")); // replace with your directory
        Process process = builder.start();
        int exitCode = process.waitFor();
        assertEquals("Return value should be 0", 0, exitCode);
    }

    private static void run_nvprof_mvn_test() throws InterruptedException, IOException {
        ProcessBuilder builder = new ProcessBuilder();
        builder.command("nvprof mvn test".split("\\s+"));
        Process process = builder.start();
        int exitCode = process.waitFor();
        assertEquals("Return value should be 0", 0, exitCode);
    }
    
}
