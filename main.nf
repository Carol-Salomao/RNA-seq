$HOSTNAME = ""
params.outdir = 'results'  

// enable required indexes to build them
params.use_Bowtie2_Index = (params.run_Sequential_Mapping == "yes") ? "yes" : ""
params.use_Bowtie_Index  = (params.run_Sequential_Mapping == "yes") ? "yes" : ""
params.use_STAR_Index    = (params.run_Sequential_Mapping == "yes" || params.run_STAR == "yes") ? "yes" : ""
params.use_Hisat2_Index  = (params.run_HISAT2 == "yes") ? "yes" : ""
params.use_RSEM_Index = (params.run_RSEM == "yes") ? "yes" : ""
params.use_Kallisto_Index = (params.run_Kallisto == "yes") ? "yes" : ""
params.nucleicAcidType = "rna"

def pathChecker(input, path, type){
	cmd = "mkdir -p check && mv ${input} check/. "
	if (!input || input.empty()){
		input = file(path).getName().toString()
		cmd = "mkdir -p check && cd check && ln -s ${path} ${input} && cd .."
		if (path.indexOf('s3:') > -1 || path.indexOf('S3:') >-1){
			recursive = (type == "folder") ? "--recursive" : ""
			cmd = "mkdir -p check && cd check && aws s3 cp ${recursive} ${path} ${workDir}/${input} && ln -s ${workDir}/${input} . && cd .."
		} else if (path.indexOf('gs:') > -1 || path.indexOf('GS:') >-1){
			if (type == "folder"){
				cmd = "mkdir -p check ${workDir}/${input} && cd check && gsutil rsync -r ${path} ${workDir}/${input} && cp -R ${workDir}/${input} . && cd .."
			} else {
				cmd = "mkdir -p check && cd check && gsutil cp ${path} ${workDir}/${input} && cp -R ${workDir}/${input} . && cd .."
			}
		} else if (path.indexOf('/') == -1){
			cmd = ""
		}
}
	return [cmd,input]
}
if (!params.run_FeatureCounts_after_STAR){params.run_FeatureCounts_after_STAR = ""} 
if (!params.run_FeatureCounts_after_Hisat2){params.run_FeatureCounts_after_Hisat2 = ""} 
if (!params.reads){params.reads = ""} 
if (!params.run_Salmon_after_STAR){params.run_Salmon_after_STAR = ""} 
if (!params.run_DESeq2_after_RSEM){params.run_DESeq2_after_RSEM = ""} 
if (!params.compare_file){params.compare_file = ""} 
if (!params.groups_file){params.groups_file = ""} 
if (!params.run_DESeq2_after_HISAT2_featurecounts){params.run_DESeq2_after_HISAT2_featurecounts = ""} 
if (!params.run_DESeq2_after_STAR_featurecounts){params.run_DESeq2_after_STAR_featurecounts = ""} 
if (!params.run_DESeq2_after_STAR_Salmon){params.run_DESeq2_after_STAR_Salmon = ""} 
if (!params.run_DESeq2_after_Kallisto){params.run_DESeq2_after_Kallisto = ""} 
if (!params.run_DESeq2_after_Salmon){params.run_DESeq2_after_Salmon = ""} 
if (!params.bam_files){params.bam_files = ""} 
if (!params.mate){params.mate = ""} 
if (!params.run_limmaVoom_after_RSEM){params.run_limmaVoom_after_RSEM = ""} 
if (!params.run_limmaVoom_after_HISAT2_featurecounts){params.run_limmaVoom_after_HISAT2_featurecounts = ""} 
if (!params.run_limmaVoom_after_STAR_featurecounts){params.run_limmaVoom_after_STAR_featurecounts = ""} 
if (!params.run_limmaVoom_after_STAR_Salmon){params.run_limmaVoom_after_STAR_Salmon = ""} 
if (!params.run_limmaVoom_after_Kallisto){params.run_limmaVoom_after_Kallisto = ""} 
if (!params.run_limmaVoom_after_Salmon){params.run_limmaVoom_after_Salmon = ""} 
if (!params.custom_additional_genome){params.custom_additional_genome = ""} 
if (!params.custom_additional_gtf){params.custom_additional_gtf = ""} 
if (!params.run_gsea_DESeq2_RSEM){params.run_gsea_DESeq2_RSEM = ""} 
if (!params.run_gsea_LimmaVoom_RSEM){params.run_gsea_LimmaVoom_RSEM = ""} 
if (!params.run_gsea_DESeq2_HISAT2_featurecounts){params.run_gsea_DESeq2_HISAT2_featurecounts = ""} 
if (!params.run_gsea_LimmaVoom_HISAT2_featurecounts){params.run_gsea_LimmaVoom_HISAT2_featurecounts = ""} 
if (!params.run_gsea_DESeq2_STAR_featurecounts){params.run_gsea_DESeq2_STAR_featurecounts = ""} 
if (!params.run_gsea_LimmaVoom_STAR_featurecounts){params.run_gsea_LimmaVoom_STAR_featurecounts = ""} 
if (!params.run_gsea_DESeq2_STAR_Salmon){params.run_gsea_DESeq2_STAR_Salmon = ""} 
if (!params.run_gsea_LimmaVoom_STAR_Salmon){params.run_gsea_LimmaVoom_STAR_Salmon = ""} 
if (!params.run_gsea_DESeq2_Kallisto){params.run_gsea_DESeq2_Kallisto = ""} 
if (!params.run_gsea_LimmaVoom_Kallisto){params.run_gsea_LimmaVoom_Kallisto = ""} 
if (!params.run_gsea_DESeq2_Salmon){params.run_gsea_DESeq2_Salmon = ""} 
if (!params.run_gsea_LimmaVoom_Salmon){params.run_gsea_LimmaVoom_Salmon = ""} 
// Stage empty file to be used as an optional input where required
ch_empty_file_1 = file("$baseDir/.emptyfiles/NO_FILE_1", hidden:true)
ch_empty_file_2 = file("$baseDir/.emptyfiles/NO_FILE_2", hidden:true)
ch_empty_file_3 = file("$baseDir/.emptyfiles/NO_FILE_3", hidden:true)
ch_empty_file_4 = file("$baseDir/.emptyfiles/NO_FILE_4", hidden:true)
ch_empty_file_5 = file("$baseDir/.emptyfiles/NO_FILE_5", hidden:true)
ch_empty_file_6 = file("$baseDir/.emptyfiles/NO_FILE_6", hidden:true)
ch_empty_file_7 = file("$baseDir/.emptyfiles/NO_FILE_7", hidden:true)
ch_empty_file_8 = file("$baseDir/.emptyfiles/NO_FILE_8", hidden:true)
ch_empty_file_9 = file("$baseDir/.emptyfiles/NO_FILE_9", hidden:true)
ch_empty_file_10 = file("$baseDir/.emptyfiles/NO_FILE_10", hidden:true)
ch_empty_file_11 = file("$baseDir/.emptyfiles/NO_FILE_11", hidden:true)
ch_empty_file_12 = file("$baseDir/.emptyfiles/NO_FILE_12", hidden:true)

Channel.value(params.run_FeatureCounts_after_STAR).set{g_179_0_g276_0}
Channel.value(params.run_FeatureCounts_after_Hisat2).set{g_188_0_g280_0}
if (params.reads){
Channel
	.fromFilePairs( params.reads , size: params.mate == "single" ? 1 : params.mate == "pair" ? 2 : params.mate == "triple" ? 3 : params.mate == "quadruple" ? 4 : -1 )
	.ifEmpty { error "Cannot find any reads matching: ${params.reads}" }
	.set{g_230_0_g_347}
 } else {  
	g_230_0_g_347 = Channel.empty()
 }

Channel.value(params.run_Salmon_after_STAR).set{g_277_4_g276_9}
Channel.value(params.run_DESeq2_after_RSEM).set{g_293_3_g292_24}
g_294_2_g292_25 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g292_24 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g306_25 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g306_24 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g305_25 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g305_24 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g302_25 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g302_24 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g303_25 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g303_24 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g304_25 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_294_2_g304_24 = params.compare_file && file(params.compare_file, type: 'any').exists() ? file(params.compare_file, type: 'any') : ch_empty_file_2
g_295_1_g251_145 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g252_145 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g253_145 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g255_145 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g274_145 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g292_25 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g292_24 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g306_25 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g306_24 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g305_25 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g305_24 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g302_25 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g302_24 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g303_25 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g303_24 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g304_25 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
g_295_1_g304_24 = params.groups_file && file(params.groups_file, type: 'any').exists() ? file(params.groups_file, type: 'any') : ch_empty_file_1
Channel.value(params.run_DESeq2_after_HISAT2_featurecounts).set{g_307_3_g306_24}
Channel.value(params.run_DESeq2_after_STAR_featurecounts).set{g_308_3_g305_24}
Channel.value(params.run_DESeq2_after_STAR_Salmon).set{g_309_3_g302_24}
Channel.value(params.run_DESeq2_after_Kallisto).set{g_310_3_g303_24}
Channel.value(params.run_DESeq2_after_Salmon).set{g_311_3_g304_24}
if (params.bam_files){
Channel
	.fromFilePairs( params.bam_files , size: params.mate == "single" ? 1 : params.mate == "pair" ? 2 : params.mate == "triple" ? 3 : params.mate == "quadruple" ? 4 : -1 )
	.ifEmpty { error "Cannot find any bamFile matching: ${params.bam_files}" }
	.set{g_348_1_g_347}
 } else {  
	g_348_1_g_347 = Channel.empty()
 }

Channel.value(params.mate).set{g_349_2_g_347}
Channel.value(params.run_limmaVoom_after_RSEM).set{g_356_3_g292_25}
Channel.value(params.run_limmaVoom_after_HISAT2_featurecounts).set{g_357_3_g306_25}
Channel.value(params.run_limmaVoom_after_STAR_featurecounts).set{g_358_3_g305_25}
Channel.value(params.run_limmaVoom_after_STAR_Salmon).set{g_359_3_g302_25}
Channel.value(params.run_limmaVoom_after_Kallisto).set{g_360_3_g303_25}
Channel.value(params.run_limmaVoom_after_Salmon).set{g_361_3_g304_25}
g_367_2_g245_58 = params.custom_additional_genome && file(params.custom_additional_genome, type: 'any').exists() ? file(params.custom_additional_genome, type: 'any') : ch_empty_file_1
g_368_3_g245_58 = params.custom_additional_gtf && file(params.custom_additional_gtf, type: 'any').exists() ? file(params.custom_additional_gtf, type: 'any') : ch_empty_file_2
Channel.value(params.run_gsea_DESeq2_RSEM).set{g_375_2_g292_33}
Channel.value(params.run_gsea_LimmaVoom_RSEM).set{g_376_2_g292_41}
Channel.value(params.run_gsea_DESeq2_HISAT2_featurecounts).set{g_389_2_g306_33}
Channel.value(params.run_gsea_LimmaVoom_HISAT2_featurecounts).set{g_390_2_g306_41}
Channel.value(params.run_gsea_DESeq2_STAR_featurecounts).set{g_391_2_g305_33}
Channel.value(params.run_gsea_LimmaVoom_STAR_featurecounts).set{g_392_2_g305_41}
Channel.value(params.run_gsea_DESeq2_STAR_Salmon).set{g_393_2_g302_33}
Channel.value(params.run_gsea_LimmaVoom_STAR_Salmon).set{g_394_2_g302_41}
Channel.value(params.run_gsea_DESeq2_Kallisto).set{g_395_2_g303_33}
Channel.value(params.run_gsea_LimmaVoom_Kallisto).set{g_396_2_g303_41}
Channel.value(params.run_gsea_DESeq2_Salmon).set{g_397_2_g304_33}
Channel.value(params.run_gsea_LimmaVoom_Salmon).set{g_398_2_g304_41}

//* @style @array:{run_name,run_parameters} @multicolumn:{run_name,run_parameters}

process Bam_Quantify_Module_HISAT2_featureCounts_Prep {

input:
 val run_featureCounts from g_188_0_g280_0

output:
 val run_params  into g280_0_run_parameters02_g280_1

when:
run_featureCounts == "yes"

script:
run_name = params.Bam_Quantify_Module_HISAT2_featureCounts_Prep.run_name
run_parameters = params.Bam_Quantify_Module_HISAT2_featureCounts_Prep.run_parameters
sense_antisense = params.Bam_Quantify_Module_HISAT2_featureCounts_Prep.sense_antisense

//define run_name and run_parameters in map item and push into run_params array
run_params = []
for (i = 0; i < run_parameters.size(); i++) {
   map = [:]
   map["run_name"] = run_name[i].replaceAll(" ","_").replaceAll(",","_").replaceAll(";","_").replaceAll("'","_").replaceAll('"',"_")
   map["run_parameters"] = run_parameters[i]
   run_params[i] = map
}
templateRunParams = run_parameters[0] ? run_parameters[0] : "-g gene_id -s 0 -Q 20 -T 2 -B -d 50 -D 1000 -C --fracOverlap 0 --minOverlap 1"
if (sense_antisense == "Yes"){
   map = [:]
   map["run_name"] = "gene_id_forward_expression"
   map["run_parameters"] = templateRunParams.replaceAll("transcript_id","gene_id").replaceAll("-s 0","-s 1").replaceAll("-s 2","-s 1")
   run_params.push(map)
   map = [:]
   map["run_name"] = "gene_id_reverse_expression"
   map["run_parameters"] = templateRunParams.replaceAll("transcript_id","gene_id").replaceAll("-s 0","-s 2").replaceAll("-s 1","-s 2")
   run_params.push(map)
   map = [:]
   map["run_name"] = "transcript_id_forward_expression"
   map["run_parameters"] = templateRunParams.replaceAll("gene_id","transcript_id").replaceAll("-s 0","-s 1").replaceAll("-s 2","-s 1")
   run_params.push(map)
   map = [:]
   map["run_name"] = "transcript_id_reverse_expression"
   map["run_parameters"] = templateRunParams.replaceAll("gene_id","transcript_id").replaceAll("-s 0","-s 2").replaceAll("-s 1","-s 2")
   run_params.push(map)
}
"""
"""

}

//* @style @array:{run_name,run_parameters} @multicolumn:{run_name,run_parameters}

process Bam_Quantify_Module_STAR_featureCounts_Prep {

input:
 val run_featureCounts from g_179_0_g276_0

output:
 val run_params  into g276_0_run_parameters02_g276_1

when:
run_featureCounts == "yes"

script:
run_name = params.Bam_Quantify_Module_STAR_featureCounts_Prep.run_name
run_parameters = params.Bam_Quantify_Module_STAR_featureCounts_Prep.run_parameters
sense_antisense = params.Bam_Quantify_Module_STAR_featureCounts_Prep.sense_antisense

//define run_name and run_parameters in map item and push into run_params array
run_params = []
for (i = 0; i < run_parameters.size(); i++) {
   map = [:]
   map["run_name"] = run_name[i].replaceAll(" ","_").replaceAll(",","_").replaceAll(";","_").replaceAll("'","_").replaceAll('"',"_")
   map["run_parameters"] = run_parameters[i]
   run_params[i] = map
}
templateRunParams = run_parameters[0] ? run_parameters[0] : "-g gene_id -s 0 -Q 20 -T 2 -B -d 50 -D 1000 -C --fracOverlap 0 --minOverlap 1"
if (sense_antisense == "Yes"){
   map = [:]
   map["run_name"] = "gene_id_forward_expression"
   map["run_parameters"] = templateRunParams.replaceAll("transcript_id","gene_id").replaceAll("-s 0","-s 1").replaceAll("-s 2","-s 1")
   run_params.push(map)
   map = [:]
   map["run_name"] = "gene_id_reverse_expression"
   map["run_parameters"] = templateRunParams.replaceAll("transcript_id","gene_id").replaceAll("-s 0","-s 2").replaceAll("-s 1","-s 2")
   run_params.push(map)
   map = [:]
   map["run_name"] = "transcript_id_forward_expression"
   map["run_parameters"] = templateRunParams.replaceAll("gene_id","transcript_id").replaceAll("-s 0","-s 1").replaceAll("-s 2","-s 1")
   run_params.push(map)
   map = [:]
   map["run_name"] = "transcript_id_reverse_expression"
   map["run_parameters"] = templateRunParams.replaceAll("gene_id","transcript_id").replaceAll("-s 0","-s 2").replaceAll("-s 1","-s 2")
   run_params.push(map)
}
"""
"""

}

g_230_0_g_347= g_230_0_g_347.ifEmpty([""]) 
g_348_1_g_347= g_348_1_g_347.ifEmpty([""]) 


if (!(params.bam_files)){
g_230_0_g_347.into{g_347_reads01_g257_28; g_347_reads00_g257_18}
g_349_2_g_347.into{g_347_mate11_g257_11; g_347_mate11_g257_16; g_347_mate11_g257_21; g_347_mate11_g257_24; g_347_mate10_g257_28; g_347_mate10_g257_31; g_347_mate11_g257_18; g_347_mate11_g257_23; g_347_mate11_g257_19; g_347_mate11_g257_20; g_347_mate11_g256_26; g_347_mate11_g256_30; g_347_mate11_g256_46; g_347_mate10_g249_14; g_347_mate10_g248_36; g_347_mate10_g268_44; g_347_mate10_g280_9; g_347_mate11_g280_1; g_347_mate10_g276_9; g_347_mate11_g276_1; g_347_mate11_g274_82; g_347_mate10_g274_131; g_347_mate12_g274_134; g_347_mate11_g255_82; g_347_mate10_g255_131; g_347_mate12_g255_134; g_347_mate11_g253_82; g_347_mate10_g253_131; g_347_mate12_g253_134; g_347_mate11_g252_82; g_347_mate10_g252_131; g_347_mate12_g252_134; g_347_mate11_g251_82; g_347_mate10_g251_131; g_347_mate12_g251_134; g_347_mate11_g264_30; g_347_mate10_g264_31; g_347_mate10_g250_26}
} else {

process bamtofastq_samtools_fastq {

input:
 set val(nameReads), file(reads) from  ( params.reads ? g_230_0_g_347 : g_230_0_g_347.first() ) 
 set val(name), file(bam) from  ( params.bam_files ? g_348_1_g_347 : g_348_1_g_347.first() ) 
 val mate from g_349_2_g_347

output:
 set val(name),file("reads/*")  into g_347_reads01_g257_28, g_347_reads00_g257_18
 env mateEnv  into g_347_mate11_g257_11, g_347_mate11_g257_16, g_347_mate11_g257_21, g_347_mate11_g257_24, g_347_mate10_g257_28, g_347_mate10_g257_31, g_347_mate11_g257_18, g_347_mate11_g257_23, g_347_mate11_g257_19, g_347_mate11_g257_20, g_347_mate11_g256_26, g_347_mate11_g256_30, g_347_mate11_g256_46, g_347_mate10_g249_14, g_347_mate10_g248_36, g_347_mate10_g268_44, g_347_mate10_g280_9, g_347_mate11_g280_1, g_347_mate10_g276_9, g_347_mate11_g276_1, g_347_mate11_g274_82, g_347_mate10_g274_131, g_347_mate12_g274_134, g_347_mate11_g255_82, g_347_mate10_g255_131, g_347_mate12_g255_134, g_347_mate11_g253_82, g_347_mate10_g253_131, g_347_mate12_g253_134, g_347_mate11_g252_82, g_347_mate10_g252_131, g_347_mate12_g252_134, g_347_mate11_g251_82, g_347_mate10_g251_131, g_347_mate12_g251_134, g_347_mate11_g264_30, g_347_mate10_g264_31, g_347_mate10_g250_26.first()

when:
params.bam_files

script:
nameAll = bam.toString()
if (nameAll.contains('.gz')) {
    file =  nameAll - '.gz'
    runGzip = "ls *.gz | xargs -i echo gzip -df {} | sh"
} else {
    file =  nameAll 
    runGzip = ''
}
"""
${runGzip}
samtools sort -n $file -o ${name}_sorted.bam
rm $file
mkdir -p reads mate
count=\$(samtools view -c -f 0x001 "${name}_sorted.bam")
# Check if the file is paired-end or not
if [ "\$count" -eq 0 ]; then
    echo "The BAM file is not paired-end."
    mateEnv=single
    samtools fastq ${name}_sorted.bam > reads/${name}.fastq
else
    echo "The BAM file is paired-end."
    mateEnv=pair
    samtools fastq -1 reads/${name}.R1.fastq -2 reads/${name}.R2.fastq -0 /dev/null -s /dev/null -n ${name}_sorted.bam
fi
gzip reads/*.fastq
"""
}
}



//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 10
}
//* platform
//* platform
//* autofill

process Adapter_Trimmer_Quality_Module_FastQC {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.(html|zip)$/) "fastqc/$filename"}
input:
 val mate from g_347_mate10_g257_28
 set val(name), file(reads) from g_347_reads01_g257_28

output:
 file '*.{html,zip}'  into g257_28_FastQCout04_g_177

when:
(params.run_FastQC && (params.run_FastQC == "yes"))

script:
"""
fastqc ${reads} 
"""
}

//* params.gtf =  ""  //* @input
//* params.genome =  ""  //* @input
//* params.commondb =  ""  //* @input
//* params.genome_source =  ""  //* @input
//* params.gtf_source =  ""  //* @input
//* params.commondb_source =  ""  //* @input @optional

def downFile(path, task){
	println workDir
    if (path.take(1).indexOf("/") == 0){
      target=path
      if (task.executor == "awsbatch" || task.executor == "google-batch") {
      	a=file(path)
    	fname = a.getName().toString()
    	target = "${workDir}/${fname}"
    	if (!file(target).exists()){
    		a.copyTo(workDir)
    	}
      }
    } else {
      a=file(path)
      fname = a.getName().toString()
      target = "${workDir}/${fname}"
      if (!file(target).exists()){
    		a.copyTo(workDir)
      } 
    }
    return target
}

def getLastName (str){
	if (str.indexOf("/") > -1){
		return  str.substring(str.lastIndexOf('/')+1,str.length())
	} 
	return ""
}

process Check_and_Build_Module_Check_Genome_GTF {


output:
 file "${newNameFasta}"  into g245_21_genome00_g245_58
 file "${newNameGtf}"  into g245_21_gtfFile10_g245_57

container 'quay.io/viascientific/pipeline_base_image:1.0'

when:
params.run_Download_Genomic_Sources == "yes"

script:
genomeSource = !file("${params.genome}").exists() ? params.genome_source : params.genome
genomeName = getLastName(genomeSource)

gtfSource = !file("${params.gtf}").exists() ? params.gtf_source : params.gtf
gtfName = getLastName(gtfSource)


newNameGtf = gtfName
newNameFasta = genomeName
if (gtfName.contains('.gz')) { newNameGtf =  newNameGtf - '.gz'  } 
if (genomeName.contains('.gz')) { newNameFasta =  newNameFasta - '.gz'  } 

runGzip = ""
if (gtfName.contains('.gz') || genomeName.contains('.gz')) {
    runGzip = "ls *.gz | xargs -i echo gzip -df {} | sh"
} 

slashCountGenome = params.genome_source.count("/")
cutDirGenome = slashCountGenome - 3;

slashCountGtf = params.gtf_source.count("/")
cutDirGtf = slashCountGtf - 3;

"""
if [ ! -e "${params.genome_source}" ] ; then
    echo "${params.genome_source} not found"
	if [[ "${params.genome_source}" =~ "s3" ]]; then
		echo "Downloading s3 path from ${params.genome_source}"
		aws s3 cp ${params.genome_source} ${workDir}/${genomeName} && ln -s ${workDir}/${genomeName} ${genomeName}
	elif [[ "${params.genome_source}" =~ "gs" ]]; then
		echo "Downloading gs path from ${params.genome_source}"
		gsutil cp  ${params.genome_source} ${workDir}/. && ln -s ${workDir}/${genomeName} ${genomeName}
	else
		echo "Downloading genome with wget"
		wget --no-check-certificate --secure-protocol=TLSv1 -l inf -nc -nH --cut-dirs=$cutDirGenome -R 'index.html*' -r --no-parent  ${params.genome_source}
	fi

else 
	ln -s ${params.genome_source} ${genomeName}
fi

if [ ! -e "${params.gtf_source}" ] ; then
    echo "${params.gtf_source} not found"
	if [[ "${params.gtf_source}" =~ "s3" ]]; then
		echo "Downloading s3 path from ${params.gtf_source}"
		aws s3 cp  ${params.gtf_source} ${workDir}/${gtfName} && ln -s ${workDir}/${gtfName} ${gtfName}
	elif [[ "${params.gtf_source}" =~ "gs" ]]; then
		echo "Downloading gs path from ${params.gtf_source}"
		gsutil cp  ${params.gtf_source} ${workDir}/. && ln -s ${workDir}/${gtfName} ${gtfName}
	else
		echo "Downloading gtf with wget"
		wget --no-check-certificate --secure-protocol=TLSv1 -l inf -nc -nH --cut-dirs=$cutDirGtf -R 'index.html*' -r --no-parent  ${params.gtf_source}
	fi

else 
	ln -s ${params.gtf_source} ${gtfName}
fi

$runGzip

"""




}


if (!(params.replace_geneID_with_geneName == "yes")){
g245_21_gtfFile10_g245_57.set{g245_57_gtfFile01_g245_58}
} else {

process Check_and_Build_Module_convert_gtf_attributes {

input:
 file gtf from g245_21_gtfFile10_g245_57

output:
 file "out/${gtf}"  into g245_57_gtfFile01_g245_58

when:
params.replace_geneID_with_geneName == "yes"

shell:
'''
#!/usr/bin/env perl 

## Replace gene_id column with gene_name column in the gtf file
## Also check if any transcript_id defined in multiple chromosomes.
system("mkdir out");

open(OUT1, ">out/!{gtf}");
open(OUT2, ">notvalid_!{gtf}");
my %transcipt;
my $file = "!{gtf}";
open IN, $file;
while( my $line = <IN>)  {
    chomp;
    @a=split("\\t",$line);
    @attr=split(";",$a[8]);
    my %h;
    for my $elem (@attr) {
        ($first, $rest) = split ' ', $elem, 2;
        $h{$first} = $rest.";";
    }
    my $geneId = "";
    my $transcript_id = "";
    if (exists $h{"gene_name"}){
        $geneId = $h{"gene_name"};
    } elsif (exists $h{"gene_id"}){
        $geneId = $h{"gene_id"};
    }
    if (exists $h{"transcript_id"}){
        $transcript_id = $h{"transcript_id"};
    } elsif (exists $h{"transcript_name"}){
        $transcript_id = $h{"transcript_name"};
    } elsif (exists $h{"gene_id"}){
        $transcript_id = $h{"gene_id"};
    }
    if ($geneId ne "" && $transcript_id ne ""){
        ## check if any transcript_id defined in multiple chromosomes.
        if (exists $transcipt{$transcript_id}){
             if ($transcipt{$transcript_id} ne $a[0]){
               print OUT2 "$transcript_id: $transcipt{$transcript_id} vs $a[0]\\n";
                next;
                }
        } else {
             $transcipt{$transcript_id} = $a[0];
        }
        $a[8]=join(" ",("gene_id",$geneId,"transcript_id",$transcript_id));
        print OUT1 join("\\t",@a), "\\n";
    }  else {
        print OUT2 "$line";
    }
}
close OUT1;
close OUT2;
close IN;
'''
}
}




if (!(params.add_sequences_to_reference == "yes")){
g245_21_genome00_g245_58.into{g245_58_genome00_g245_52; g245_58_genome01_g245_54}
g245_57_gtfFile01_g245_58.into{g245_58_gtfFile10_g245_53; g245_58_gtfFile10_g245_54}
} else {

process Check_and_Build_Module_Add_custom_seq_to_genome_gtf {

input:
 file genome from g245_21_genome00_g245_58
 file gtf from g245_57_gtfFile01_g245_58
 file custom_fasta from g_367_2_g245_58
 file custom_gtf from g_368_3_g245_58

output:
 file "${genomeName}_custom.fa"  into g245_58_genome00_g245_52, g245_58_genome01_g245_54
 file "${gtfName}_custom_sorted.gtf"  into g245_58_gtfFile10_g245_53, g245_58_gtfFile10_g245_54

container 'quay.io/viascientific/custom_sequence_to_genome_gtf:1.0'

when:
params.add_sequences_to_reference == "yes"

script:
genomeName = genome.baseName
gtfName = gtf.baseName
is_custom_genome_exists = custom_fasta.name.startsWith('NO_FILE') ? "False" : "True" 
is_custom_gtf_exists = custom_gtf.name.startsWith('NO_FILE') ? "False" : "True" 
"""
#!/usr/bin/env python 
import requests
import os
import pandas as pd
import re
import urllib
from Bio import SeqIO

def add_to_fasta(seq, sqid, out_name):
	new_line = '>' + sqid + '\\n' + seq + '\\n'
	with open(out_name + '.fa', 'a') as f:
		f.write(new_line)

def createCustomGtfFromFasta(fastaFile, outCustomGtfFile):

    fasta_sequences = SeqIO.parse(open(fastaFile),'fasta')
    with open(outCustomGtfFile, "w") as out_file:
        for fasta in fasta_sequences:
            name, sequence = fasta.id, str(fasta.seq)
            last = len(sequence)
            line1 = "{gene}\\tKNOWN\\tgene\\t{first}\\t{last}\\t.\\t+\\t.\\tgene_id \\"{gene}\\"; gene_version \\"1\\"; gene_type \\"protein_coding\\"; gene_source \\"KNOWN\\"; gene_name \\"{gene}\\"; gene_biotype \\"protein_coding\\"; gene_status \\"KNOWN\\"; level 1;".format(gene=name, first="1", last=last)
            line2 = "{gene}\\tKNOWN\\ttranscript\\t{first}\\t{last}\\t.\\t+\\t.\\tgene_id \\"{gene}\\"; gene_version \\"1\\"; transcript_id \\"{gene}_trans\\"; transcript_version \\"1\\"; gene_type \\"protein_coding\\"; gene_source \\"KNOWN\\"; transcript_source \\"KNOWN\\"; gene_status \\"KNOWN\\"; gene_name \\"{gene}\\"; gene_biotype \\"protein_coding\\"; transcript_type \\"protein_coding\\"; transcript_status \\"KNOWN\\"; transcript_name \\"{gene}_1\\"; level 1; tag \\"basic\\"; transcript_biotype \\"protein_coding\\"; transcript_support_level \\"1\\";".format(gene=name, first="1", last=last)
            line3 = "{gene}\\tKNOWN\\texon\\t{first}\\t{last}\\t.\\t+\\t.\\tgene_id \\"{gene}\\"; gene_version \\"1\\"; transcript_id \\"{gene}_trans\\"; transcript_version \\"1\\"; exon_number 1; gene_type \\"protein_coding\\"; gene_source \\"KNOWN\\"; transcript_source \\"KNOWN\\"; gene_status \\"KNOWN\\"; gene_name \\"{gene}\\"; gene_biotype \\"protein_coding\\"; transcript_type \\"protein_coding\\"; transcript_status \\"KNOWN\\"; transcript_biotype \\"protein_coding\\"; transcript_name \\"{gene}_1\\"; exon_number 1; exon_id \\"{gene}.1\\"; level 1; tag \\"basic\\"; transcript_support_level \\"1\\";".format(gene=name, first="1", last=last)
            out_file.write("{}\\n{}\\n{}\\n".format(line1, line2, line3))

	
os.system('cp ${genomeName}.fa ${genomeName}_custom.fa')  
os.system('cp ${gtfName}.gtf ${gtfName}_custom.gtf')  

if ${is_custom_genome_exists}:
	os.system("tr -d '\\r' < ${custom_fasta} > ${custom_fasta}_tmp && rm ${custom_fasta} && mv ${custom_fasta}_tmp ${custom_fasta}")
	os.system('cat ${custom_fasta} >> ${genomeName}_custom.fa')
	if ${is_custom_gtf_exists}:
		os.system("tr -d '\\r' < ${custom_gtf} > ${custom_gtf}_tmp && rm ${custom_gtf} && mv ${custom_gtf}_tmp ${custom_gtf}")
		os.system("mv ${custom_gtf} ${custom_fasta}.gtf")
	else:
		createCustomGtfFromFasta("${custom_fasta}", "${custom_fasta}.gtf")
	os.system('cat ${custom_fasta}.gtf >> ${gtfName}_custom.gtf')

	
os.system('samtools faidx ${genomeName}_custom.fa')
os.system('igvtools sort ${gtfName}_custom.gtf ${gtfName}_custom_sorted.gtf')
os.system('igvtools index ${gtfName}_custom_sorted.gtf')

"""
}
}


//* params.gtf2bed_path =  ""  //* @input
//* params.bed =  ""  //* @input

process Check_and_Build_Module_Check_BED12 {

input:
 file gtf from g245_58_gtfFile10_g245_53

output:
 file "${gtfName}.bed"  into g245_53_bed03_g245_54

when:
params.run_Download_Genomic_Sources == "yes"

script:
gtfName  = gtf.baseName
beddir = ""
if (params.bed.indexOf('/') > -1){
	beddir  = params.bed.substring(0, params.bed.lastIndexOf('/')) 
}
"""

if [ ! -e "${params.bed}" ] ; then
    echo "${params.bed} not found"
    perl ${params.gtf2bed_path} $gtf > ${gtfName}.bed
else 
	cp -n ${params.bed} ${gtfName}.bed
fi
if [ "${beddir}" != "" ] ; then
	mkdir -p ${beddir}
	cp -n ${gtfName}.bed ${params.bed} 
fi
"""




}

//* params.gtf2bed_path =  ""  //* @input
//* params.genome_sizes =  ""  //* @input

process Check_and_Build_Module_Check_chrom_sizes_and_index {

input:
 file genome from g245_58_genome00_g245_52

output:
 file "${genomeName}.chrom.sizes"  into g245_52_genomeSizes02_g245_54

when:
params.run_Download_Genomic_Sources == "yes"

script:
genomeName  = genome.baseName
genome_sizes_dir = ""
if (params.genome_sizes.indexOf('/') > -1){
	genome_sizes_dir  = params.genome_sizes.substring(0, params.genome_sizes.lastIndexOf('/')) 
}

"""
if [ ! -e "${params.genome_sizes}" ] ; then
    echo "${params.genome_sizes} not found"
    cat ${genome} | awk '\$0 ~ ">" {print c; c=0;printf substr(\$1,2,100) "\\t"; } \$0 !~ ">" {c+=length(\$0);} END { print c; }' > ${genomeName}.chrom.sizes
    ##clean first empty line
    sed -i '1{/^\$/d}' ${genomeName}.chrom.sizes
    if [ "${genome_sizes_dir}" != "" ] ; then
    	mkdir -p ${genome_sizes_dir}
		cp -n ${genomeName}.chrom.sizes ${params.genome_sizes} 
	fi
else 
	cp ${params.genome_sizes} ${genomeName}.chrom.sizes
fi

"""




}

g245_58_gtfFile10_g245_54= g245_58_gtfFile10_g245_54.ifEmpty([""]) 
g245_58_genome01_g245_54= g245_58_genome01_g245_54.ifEmpty([""]) 
g245_52_genomeSizes02_g245_54= g245_52_genomeSizes02_g245_54.ifEmpty([""]) 
g245_53_bed03_g245_54= g245_53_bed03_g245_54.ifEmpty([""]) 


process Check_and_Build_Module_check_files {

input:
 file gtf from g245_58_gtfFile10_g245_54
 file genome from g245_58_genome01_g245_54
 file genomeSizes from g245_52_genomeSizes02_g245_54
 file bed from g245_53_bed03_g245_54

output:
 file "*/${gtf2}" optional true  into g245_54_gtfFile01_g256_47, g245_54_gtfFile01_g249_16, g245_54_gtfFile01_g248_38, g245_54_gtfFile01_g248_39, g245_54_gtfFile01_g248_32, g245_54_gtfFile03_g248_36, g245_54_gtfFile01_g268_42, g245_54_gtfFile03_g268_44, g245_54_gtfFile01_g268_47, g245_54_gtfFile01_g268_48, g245_54_gtfFile01_g276_9, g245_54_gtfFile01_g276_14, g245_54_gtfFile01_g276_15, g245_54_gtfFile03_g276_1, g245_54_gtfFile01_g280_9, g245_54_gtfFile01_g280_14, g245_54_gtfFile01_g280_15, g245_54_gtfFile03_g280_1, g245_54_gtfFile01_g264_21, g245_54_gtfFile01_g250_31
 file "*/${genome2}" optional true  into g245_54_genome10_g256_47, g245_54_genome10_g249_16, g245_54_genome10_g248_32, g245_54_genome15_g248_36, g245_54_genome10_g268_42, g245_54_genome15_g268_44, g245_54_genome12_g276_9, g245_54_genome12_g280_9, g245_54_genome10_g264_21, g245_54_genome10_g250_31
 file "*/${genomeSizes2}" optional true  into g245_54_genomeSizes24_g248_36, g245_54_genomeSizes24_g268_44, g245_54_genomeSizes22_g274_131, g245_54_genomeSizes21_g274_142, g245_54_genomeSizes22_g255_131, g245_54_genomeSizes21_g255_142, g245_54_genomeSizes22_g253_131, g245_54_genomeSizes21_g253_142, g245_54_genomeSizes22_g252_131, g245_54_genomeSizes21_g252_142, g245_54_genomeSizes22_g251_131, g245_54_genomeSizes21_g251_142
 file "*/${bed2}" optional true  into g245_54_bed31_g274_134, g245_54_bed31_g255_134, g245_54_bed31_g253_134, g245_54_bed31_g252_134, g245_54_bed31_g251_134

container 'quay.io/viascientific/pipeline_base_image:1.0'

script:
(cmd1, gtf2) = pathChecker(gtf, params.gtf, "file")
(cmd2, genome2) = pathChecker(genome, params.genome, "file")
(cmd3, genomeSizes2) = pathChecker(genomeSizes, params.genome_sizes, "file")
(cmd4, bed2) = pathChecker(bed, params.bed, "file")
"""
$cmd1
$cmd2
$cmd3
$cmd4
"""
}

build_RSEM_index = params.RSEM_module_Check_Build_Rsem_Index.build_RSEM_index

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 16
    $MEMORY = 100
}
//* platform
if ($HOSTNAME == "hpc.umassmed.edu"){
    $CPU  = 16
    $MEMORY = 12
}
//* platform
//* autofill

process RSEM_module_Check_Build_Rsem_Index {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${index}$/) "rsem_index/$filename"}
input:
 file genome from g245_54_genome10_g250_31
 file gtf from g245_54_gtfFile01_g250_31

output:
 file "${index}"  into g250_31_rsemIndex00_g250_32

container "quay.io/viascientific/rsem:1.0"

when:
build_RSEM_index == true && ((params.run_RSEM && (params.run_RSEM == "yes")) || !params.run_RSEM)

script:
transcript_to_gene_map = params.RSEM_module_Check_Build_Rsem_Index.transcript_to_gene_map
RSEM_build_parameters = params.RSEM_module_Check_Build_Rsem_Index.RSEM_build_parameters

transcript_to_gene_mapText = ""
if (transcript_to_gene_map?.trim()){
    transcript_to_gene_mapText = "--transcript-to-gene-map " + transcript_to_gene_map
}
RSEM_reference_type = params.RSEM_module_RSEM.RSEM_reference_type
basename = genome.baseName

indexType = ""
index = ""
index_dir = ""
if (RSEM_reference_type == 'bowtie'){
	indexType = "--bowtie "
	index = "RSEM_ref_Bowtie" 
	if (params.rsem_ref_using_bowtie_index.indexOf('/') > -1 && params.rsem_ref_using_bowtie_index.indexOf('s3://') < 0){
		index_dir = params.rsem_ref_using_bowtie_index
	}
} else if (RSEM_reference_type == 'bowtie2'){
	indexType = "--bowtie2 "
	index = "RSEM_ref_Bowtie2" 
	if (params.rsem_ref_using_bowtie2_index.indexOf('/') > -1 && params.rsem_ref_using_bowtie2_index.indexOf('s3://') < 0){
		index_dir = params.rsem_ref_using_bowtie2_index
	}
} else if (RSEM_reference_type == 'star'){
	indexType = "--star "
	index = "RSEM_ref_STAR" 
	if (params.rsem_ref_using_star_index.indexOf('/') > -1 && params.rsem_ref_using_star_index.indexOf('s3://') < 0){
		index_dir = params.rsem_ref_using_star_index
	}
}

"""
if [ ! -e "${index_dir}/${basename}.ti" ] ; then
    echo "${index_dir}/${basename}.ti RSEM index not found"
    
    mkdir -p $index && mv $genome $gtf $index/. && cd $index
    rsem-prepare-reference ${RSEM_build_parameters} --gtf ${gtf} ${transcript_to_gene_mapText} ${indexType} ${genome} ${basename}
    cd ..
else 
	ln -s ${index_dir} $index
fi
"""

}

g250_31_rsemIndex00_g250_32= g250_31_rsemIndex00_g250_32.ifEmpty([""]) 


if (!((params.run_RSEM && (params.run_RSEM == "yes")) || !params.run_RSEM)){
g250_31_rsemIndex00_g250_32.set{g250_32_rsemIndex02_g250_26}
} else {

process RSEM_module_check_RSEM_files {

input:
 file rsem from g250_31_rsemIndex00_g250_32

output:
 file "*/${rsem2}" optional true  into g250_32_rsemIndex02_g250_26

container 'quay.io/viascientific/pipeline_base_image:1.0'

when:
(params.run_RSEM && (params.run_RSEM == "yes")) || !params.run_RSEM

script:
RSEM_reference_type = ''
if (params.RSEM && params.RSEM.RSEM_reference_type){
	RSEM_reference_type = params.RSEM.RSEM_reference_type		
} else if (params.RSEM_module_RSEM && params.RSEM_module_RSEM.RSEM_reference_type){
	RSEM_reference_type = params.RSEM_module_RSEM.RSEM_reference_type
}

if (RSEM_reference_type == 'bowtie'){
	systemInput = params.rsem_ref_using_bowtie_index
} else if (RSEM_reference_type == 'bowtie2'){
	systemInput = params.rsem_ref_using_bowtie2_index
} else if (RSEM_reference_type == 'star'){
	systemInput = params.rsem_ref_using_star_index
}	

	
(cmd, rsem2) = pathChecker(rsem, systemInput, "folder")
"""
$cmd
"""
}
}


build_STAR_index = params.STAR_Module_Check_Build_STAR_Index.build_STAR_index

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 150
}
//* platform
if ($HOSTNAME == "hpc.umassmed.edu"){
    $CPU  = 5
    $MEMORY = 30
}
//* platform
//* autofill

process STAR_Module_Check_Build_STAR_Index {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /STARIndex$/) "star_index/$filename"}
input:
 file genome from g245_54_genome10_g264_21
 file gtf from g245_54_gtfFile01_g264_21

output:
 file "STARIndex"  into g264_21_starIndex00_g264_26

container "quay.io/viascientific/rsem:1.0"

when:
build_STAR_index == true && ((params.run_STAR && (params.run_STAR == "yes")) || !params.run_STAR)

script:
star_build_parameters = params.STAR_Module_Check_Build_STAR_Index.star_build_parameters
newDirName = "STARIndex" 
"""
if [ ! -e "${params.star_index}/SA" ] ; then
    echo "STAR index not found"
    mkdir -p $newDirName 
    STAR --runMode genomeGenerate ${star_build_parameters} --genomeDir $newDirName --genomeFastaFiles ${genome} --sjdbGTFfile ${gtf}
else 
	ln -s ${params.star_index} STARIndex
fi

"""





}

g264_21_starIndex00_g264_26= g264_21_starIndex00_g264_26.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 10
}
//* platform
//* platform
//* autofill
if (!((params.run_STAR && (params.run_STAR == "yes")) || !params.run_STAR)){
g264_21_starIndex00_g264_26.set{g264_26_starIndex02_g264_31}
} else {


process STAR_Module_check_STAR_files {

input:
 file star from g264_21_starIndex00_g264_26

output:
 file "*/${star2}" optional true  into g264_26_starIndex02_g264_31

container 'quay.io/viascientific/pipeline_base_image:1.0'

when:
(params.run_STAR && (params.run_STAR == "yes")) || !params.run_STAR

script:
(cmd, star2) = pathChecker(star, params.star_index, "folder")
"""
$cmd
"""
}
}


//* params.salmon_index =  ""  //* @input
//* params.genome_sizes =  ""  //* @input
//* params.gtf =  ""  //* @input
//* @style @multicolumn:{fragment_length,standard_deviation} @condition:{single_or_paired_end_reads="single", fragment_length,standard_deviation}, {single_or_paired_end_reads="pair"}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 4
    $MEMORY = 20 
}
//* platform
//* platform
//* autofill


process Bam_Quantify_Module_HISAT2_salmon_bam_quant {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /salmon_${name}$/) "salmon_bam_count_hisat2/$filename"}
input:
 val mate from g_347_mate10_g280_9
 file gtf from g245_54_gtfFile01_g280_9
 file genome from g245_54_genome12_g280_9

output:
 file "salmon_${name}"  into g280_9_outputDir00_g280_14

container 'quay.io/viascientific/salmon:1.0'

when:
runSalmonBamCount == "yes"

script:
salmon_parameters = params.Bam_Quantify_Module_HISAT2_salmon_bam_quant.salmon_parameters
libType = params.Bam_Quantify_Module_HISAT2_salmon_bam_quant.libType
//. strandedness = meta.single_end ? 'U' : 'IU'
//     if (meta.strandedness == 'forward') {
//         strandedness = meta.single_end ? 'SF' : 'ISF'
//     } else if (meta.strandedness == 'reverse') {
//         strandedness = meta.single_end ? 'SR' : 'ISR'
"""
# filter_gtf_for_genes_in_genome.py --gtf ${gtf} --fasta ${genome} -o genome_filtered_genes.gtf
gffread -F -w transcripts_raw.fa -g ${genome} ${gtf}
cut -d ' ' -f1 transcripts_raw.fa > transcripts.fa
salmon quant -t transcripts.fa --threads $task.cpus --libType=$libType -a $bam  $salmon_parameters  -o salmon_${name} 



if [ -f salmon_${name}/quant.sf ]; then
  mv salmon_${name}/quant.sf  salmon_${name}/abundance_isoforms.tsv
fi


"""

}

//* params.gtf =  ""  //* @input

//* autofill
//* platform
//* platform
//* autofill

process Bam_Quantify_Module_HISAT2_Salmon_transcript_to_gene_count {

input:
 file outDir from g280_9_outputDir00_g280_14
 file gtf from g245_54_gtfFile01_g280_14

output:
 file newoutDir  into g280_14_outputDir00_g280_15

shell:
newoutDir = "genes_" + outDir
'''
#!/usr/bin/env perl
use strict;
use Getopt::Long;
use IO::File;
use Data::Dumper;

my $gtf_file = "!{gtf}";
my $transcript_matrix_in = "!{outDir}/abundance_isoforms.tsv";
my $transcript_matrix_out = "!{outDir}/abundance_genes.tsv";
open(IN, "<$gtf_file") or die "Can't open $gtf_file.\\n";
my %all_genes; # save gene_id of transcript_id
while(<IN>){
  next if(/^##/); #ignore header
  chomp;
  my %attribs = ();
  my ($chr, $source, $type, $start, $end, $score,
    $strand, $phase, $attributes) = split("\\t");
  my @add_attributes = split(";", $attributes);
  # store ids and additional information in second hash
  foreach my $attr ( @add_attributes ) {
     next unless $attr =~ /^\\s*(.+)\\s(.+)$/;
     my $c_type  = $1;
     my $c_value = $2;
     $c_value =~ s/\\"//g;
     if($c_type  && $c_value){
       if(!exists($attribs{$c_type})){
         $attribs{$c_type} = [];
       }
       push(@{ $attribs{$c_type} }, $c_value);
     }
  }
  #work with the information from the two hashes...
  if(exists($attribs{'transcript_id'}->[0]) && exists($attribs{'gene_id'}->[0])){
    if(!exists($all_genes{$attribs{'transcript_id'}->[0]})){
        $all_genes{$attribs{'transcript_id'}->[0]} = $attribs{'gene_id'}->[0];
    }
  } 
}


# print Dumper \\%all_genes;

#Parse the salmon input file, determine gene IDs for each transcript, and calculate sum TPM values
my %gene_exp;
my %gene_length;
my %samples;
my $ki_fh = IO::File->new($transcript_matrix_in, 'r');
my $header = '';
my $h = 0;
while (my $ki_line = $ki_fh->getline) {
  $h++;
  chomp($ki_line);
  my @ki_entry = split("\\t", $ki_line);
  my $s = 0;
  if ($h == 1){
    $header = $ki_line;
    my $first_col = shift @ki_entry;
    my $second_col = shift @ki_entry;
    foreach my $sample (@ki_entry){
      $s++;
      $samples{$s}{name} = $sample;
    }
    next;
  }
  my $trans_id = shift @ki_entry;
  my $length = shift @ki_entry;
  my $gene_id;
  if ($all_genes{$trans_id}){
    $gene_id = $all_genes{$trans_id};
  }elsif($trans_id =~ /ERCC/){
    $gene_id = $trans_id;
  }else{
    print "\\n\\nCould not identify gene id from trans id: $trans_id\\n\\n";
  }

  $s = 0;
  foreach my $value (@ki_entry){
    $s++;
    $gene_exp{$gene_id}{$s} += $value;
  }
  if ($gene_length{$gene_id}){
    $gene_length{$gene_id} = $length if ($length > $gene_length{$gene_id});
  }else{
    $gene_length{$gene_id} = $length;
  }

}
$ki_fh->close;

my $ko_fh = IO::File->new($transcript_matrix_out, 'w');
unless ($ko_fh) { die('Failed to open file: '. $transcript_matrix_out); }

print $ko_fh "$header\\n";
foreach my $gene_id (sort {$a cmp $b} keys %gene_exp){
  print $ko_fh "$gene_id\\t$gene_length{$gene_id}\\t";
  my @vals;
  foreach my $s (sort {$a <=> $b} keys %samples){
     push(@vals, $gene_exp{$gene_id}{$s});
  }
  my $val_string = join("\\t", @vals);
  print $ko_fh "$val_string\\n";
}


$ko_fh->close;
if (checkFile("!{outDir}")){
	rename ("!{outDir}", "!{newoutDir}");
}

sub checkFile {
    my ($file) = @_;
    print "$file\\n";
    return 1 if ( -e $file );
    return 0;
}

'''
}

//* params.gtf =  ""  //* @input

//* autofill
//* platform
//* platform
//* autofill

process Bam_Quantify_Module_HISAT2_Salmon_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "salmon_bam_count_hisat2_summary/$filename"}
input:
 file salmonOut from g280_14_outputDir00_g280_15.collect()
 file gtf from g245_54_gtfFile01_g280_15

output:
 file "*.tsv"  into g280_15_outputFile00

shell:
'''
#!/usr/bin/env perl
use Data::Dumper;
use strict;

### Parse gtf file
my $gtf_file = "!{gtf}";
open(IN, "<$gtf_file") or die "Can't open $gtf_file.\\n";
my %all_genes; # save gene_id of transcript_id
my %all_trans; # map transcript_id of genes
while(<IN>){
  next if(/^##/); #ignore header
  chomp;
  my %attribs = ();
  my ($chr, $source, $type, $start, $end, $score,
    $strand, $phase, $attributes) = split("\\t");
  my @add_attributes = split(";", $attributes);
  # store ids and additional information in second hash
  foreach my $attr ( @add_attributes ) {
     next unless $attr =~ /^\\s*(.+)\\s(.+)$/;
     my $c_type  = $1;
     my $c_value = $2;
     $c_value =~ s/\\"//g;
     if($c_type  && $c_value){
       if(!exists($attribs{$c_type})){
         $attribs{$c_type} = [];
       }
       push(@{ $attribs{$c_type} }, $c_value);
     }
  }
  #work with the information from the two hashes...
  if(exists($attribs{'transcript_id'}->[0]) && exists($attribs{'gene_id'}->[0])){
    if(!exists($all_genes{$attribs{'transcript_id'}->[0]})){
        $all_genes{$attribs{'transcript_id'}->[0]} = $attribs{'gene_id'}->[0];
    }
    if(!exists($all_trans{$attribs{'gene_id'}->[0]})){
        $all_trans{$attribs{'gene_id'}->[0]} = $attribs{'transcript_id'}->[0];
    } else {
    	if (index($all_trans{$attribs{'gene_id'}->[0]}, $attribs{'transcript_id'}->[0]) == -1) {
			$all_trans{$attribs{'gene_id'}->[0]} = $all_trans{$attribs{'gene_id'}->[0]} . "," .$attribs{'transcript_id'}->[0];
		}
    	
    }
  } 
}


print Dumper \\%all_trans;



#### Create summary table

my %tf = (
        expected_count => 4,
        tpm => 3
    );

my $indir = $ENV{'PWD'};
my $outdir = $ENV{'PWD'};

my @gene_iso_ar = ("genes","isoforms");
my @tpm_fpkm_expectedCount_ar = ("expected_count", "tpm");
for(my $l = 0; $l <= $#gene_iso_ar; $l++) {
    my $gene_iso = $gene_iso_ar[$l];
    for(my $ll = 0; $ll <= $#tpm_fpkm_expectedCount_ar; $ll++) {
        my $tpm_fpkm_expectedCount = $tpm_fpkm_expectedCount_ar[$ll];

        opendir D, $indir or die "Could not open $indir\\n";
        my @alndirs = sort { $a cmp $b } grep /^genes_salmon_/, readdir(D);
        closedir D;
    
        my @a=();
        my %b=();
        my %c=();
        my $i=0;
        foreach my $d (@alndirs){ 
            my $dir = "${indir}/$d";
            print $d."\\n";
            my $libname=$d;
            $libname=~s/genes_salmon_//;
            $i++;
            $a[$i]=$libname;
            open IN,"${dir}/abundance_${gene_iso}.tsv";
            $_=<IN>;
            while(<IN>)
            {
                my @v=split; 
                # $v[0] -> transcript_id
                # $all_genes{$v[0]} -> $gene_id
                if ($gene_iso eq "isoforms"){
                	$c{$v[0]}=$all_genes{$v[0]};
                } elsif ($gene_iso eq "genes"){
                	$c{$v[0]}=$all_trans{$v[0]};
                } 
                $b{$v[0]}{$i}=$v[$tf{$tpm_fpkm_expectedCount}];
                 
            }
            close IN;
        }
        my $outfile="${indir}/"."$gene_iso"."_expression_"."$tpm_fpkm_expectedCount".".tsv";
        open OUT, ">$outfile";
        if ($gene_iso ne "isoforms") {
            print OUT "gene\\ttranscript";
        } else {
            print OUT "transcript\\tgene";
        }
    
        for(my $j=1;$j<=$i;$j++) {
            print OUT "\\t$a[$j]";
        }
        print OUT "\\n";
    
        foreach my $key (keys %b) {
            print OUT "$key\\t$c{$key}";
            for(my $j=1;$j<=$i;$j++){
                print OUT "\\t$b{$key}{$j}";
            }
            print OUT "\\n";
        }
        close OUT;
    }
}

'''
}

build_Salmon_index = params.Salmon_module_Check_Build_Salmon_Index.build_Salmon_index

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 30
}
//* platform
//* platform
//* autofill

process Salmon_module_Check_Build_Salmon_Index {

input:
 file genome from g245_54_genome10_g268_42
 file gtf from g245_54_gtfFile01_g268_42

output:
 file "$index"  into g268_42_salmon_index00_g268_43

container 'quay.io/viascientific/salmon:1.0'

when:
build_Salmon_index == true && ((params.run_Salmon && (params.run_Salmon == "yes")) || !params.run_Salmon)

script:
index_dir = ""
if (params.salmon_index.indexOf('/') > -1 && params.salmon_index.indexOf('s3://') < 0){
	index_dir = file(params.salmon_index).getParent()
}
index = "SalmonIndex" 
salmon_index_parameters = params.Salmon_module_Check_Build_Salmon_Index.salmon_index_parameters
"""
if [ ! -e "${index_dir}/sa.bin" ] ; then
    echo "${index_dir}/sa.bin Salmon index not found"
    
    # filter_gtf_for_genes_in_genome.py --gtf ${gtf} --fasta ${genome} -o genome_filtered_genes.gtf
    gffread -F -w transcripts_raw.fa -g ${genome} ${gtf}
    cut -d ' ' -f1 transcripts_raw.fa > transcripts.fa

    grep '^>' $genome | cut -d ' ' -f 1 > decoys.txt
    sed -i.bak -e 's/>//g' decoys.txt
    cat transcripts.fa $genome > gentrome.fa
    salmon  index --threads $task.cpus -t gentrome.fa -d decoys.txt $salmon_index_parameters -i ${index}

else 
	ln -s ${index_dir} $index
fi
"""




}

g268_42_salmon_index00_g268_43= g268_42_salmon_index00_g268_43.ifEmpty([""]) 


if (!((params.run_Salmon && (params.run_Salmon == "yes")) || !params.run_Salmon)){
g268_42_salmon_index00_g268_43.set{g268_43_salmon_index02_g268_44}
} else {

process Salmon_module_check_Salmon_files {

input:
 file salmon from g268_42_salmon_index00_g268_43

output:
 file "*/${salmon2}" optional true  into g268_43_salmon_index02_g268_44

container 'quay.io/viascientific/pipeline_base_image:1.0'

when:
(params.run_Salmon && (params.run_Salmon == "yes")) || !params.run_Salmon

script:
(cmd, salmon2) = pathChecker(salmon, params.salmon_index, "folder")
"""
$cmd
"""
}
}


build_Kallisto_index = params.Kallisto_module_Check_Build_Kallisto_Index.build_Kallisto_index

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 30
}
//* platform
//* platform
//* autofill

process Kallisto_module_Check_Build_Kallisto_Index {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /$index\/.*.idx$/) "kallisto_index/$filename"}
input:
 file genome from g245_54_genome10_g248_32
 file gtf from g245_54_gtfFile01_g248_32

output:
 file "$index/*.idx"  into g248_32_kallisto_index00_g248_31

when:
build_Kallisto_index == true && ((params.run_Kallisto && (params.run_Kallisto == "yes")) || !params.run_Kallisto)

script:
index_dir = ""
if (params.kallisto_index.indexOf('/') > -1 && params.kallisto_index.indexOf('s3://') < 0){
	index_dir = file(params.kallisto_index).getParent()
}
index = "KallistoIndex" 

"""
if [ ! -e "${index_dir}/transcripts.idx" ] ; then
    echo "${index_dir}/transcripts.idx Kallisto index not found"
    
    mkdir -p $index && mv $genome $gtf $index/. && cd $index
    filter_gtf_for_genes_in_genome.py --gtf ${gtf} --fasta ${genome} -o genome_filtered_genes.gtf
    gawk '( \$3 ~ /gene/ )' genome_filtered_genes.gtf > new.gtf
    gawk '( \$3 ~ /transcript/ )' genome_filtered_genes.gtf >> new.gtf
    gawk '( \$3 ~ /exon/ && \$7 ~ /+/ )' genome_filtered_genes.gtf | sort -k1,1 -k4,4n >> new.gtf
    gawk '( \$3 ~ /exon/ && \$7 ~ /-/ )' genome_filtered_genes.gtf | sort -k1,1 -k4,4nr >> new.gtf

    
    gffread -F -w transcripts_raw.fa -g ${genome} new.gtf
    cut -d ' ' -f1 transcripts_raw.fa > transcripts.fa
    gzip transcripts.fa
    kallisto index --make-unique -i transcripts.idx transcripts.fa.gz
    
else 
	ln -s ${index_dir} $index
fi
"""



}

g248_32_kallisto_index00_g248_31= g248_32_kallisto_index00_g248_31.ifEmpty([""]) 


if (!((params.run_Kallisto && (params.run_Kallisto == "yes")) || !params.run_Kallisto)){
g248_32_kallisto_index00_g248_31.set{g248_31_kallisto_index02_g248_36}
} else {

process Kallisto_module_check_kallisto_files {

input:
 file kallisto from g248_32_kallisto_index00_g248_31

output:
 file "*/${kallisto2}" optional true  into g248_31_kallisto_index02_g248_36

container 'quay.io/viascientific/pipeline_base_image:1.0'

when:
(params.run_Kallisto && (params.run_Kallisto == "yes")) || !params.run_Kallisto

script:
(cmd, kallisto2) = pathChecker(kallisto, params.kallisto_index, "file")
"""
$cmd
"""
}
}


build_Hisat2_index = params.HISAT2_Module_Check_Build_Hisat2_Index.build_Hisat2_index
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 200
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 3000
    $CPU  = 5
    $MEMORY = 200
    $QUEUE = "long"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 3000
    $CPU  = 5
    $MEMORY = 50
    $QUEUE = "long"
}
//* platform
//* autofill

process HISAT2_Module_Check_Build_Hisat2_Index {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /$index$/) "hisat2_index/$filename"}
input:
 file genome from g245_54_genome10_g249_16
 file gtf from g245_54_gtfFile01_g249_16

output:
 file "$index"  into g249_16_hisat2Index00_g249_15

when:
build_Hisat2_index == true && ((params.run_HISAT2 && (params.run_HISAT2 == "yes")) || !params.run_HISAT2)

script:
hisat2_build_parameters = params.HISAT2_Module_Check_Build_Hisat2_Index.hisat2_build_parameters
basename = genome.baseName
basenameGTF = gtf.baseName
index_dir = ""
if (params.hisat2_index.indexOf('/') > -1 && params.hisat2_index.indexOf('s3://') < 0){
	index_dir  = file(params.hisat2_index).getParent()
}
index = "Hisat2Index" 

extract_splice_sites = "hisat2_extract_splice_sites.py ${gtf} > ${basenameGTF}.hisat2_splice_sites.txt"
extract_exons = "hisat2_extract_exons.py ${gtf}> ${basenameGTF}.hisat2_exons.txt"
ss = "--ss ${basenameGTF}.hisat2_splice_sites.txt"
exon = "--exon ${basenameGTF}.hisat2_exons.txt"

"""
if [ ! -e "${index_dir}/${basename}.8.ht2" ] ; then
    echo "${index_dir}/${basename}.8.ht2 Hisat2 index not found"
    
    mkdir -p $index && mv $genome $gtf $index/. && cd $index
    $extract_splice_sites
    $extract_exons
    hisat2-build ${hisat2_build_parameters} $ss $exon ${genome} ${basename}
else 
	ln -s ${index_dir} $index
fi
"""




}

g249_16_hisat2Index00_g249_15= g249_16_hisat2Index00_g249_15.ifEmpty([""]) 


if (!((params.run_HISAT2 && (params.run_HISAT2 == "yes")) || !params.run_HISAT2)){
g249_16_hisat2Index00_g249_15.set{g249_15_hisat2Index02_g249_14}
} else {

process HISAT2_Module_check_Hisat2_files {

input:
 file hisat2 from g249_16_hisat2Index00_g249_15

output:
 file "*/${hisat2new}" optional true  into g249_15_hisat2Index02_g249_14

container 'quay.io/viascientific/pipeline_base_image:1.0'

when:
(params.run_HISAT2 && (params.run_HISAT2 == "yes")) || !params.run_HISAT2

script:
(cmd, hisat2new) = pathChecker(hisat2, params.hisat2_index, "folder")
"""
$cmd
"""
}
}


download_build_sequential_mapping_indexes = params.Sequential_Mapping_Module_Download_build_sequential_mapping_indexes.download_build_sequential_mapping_indexes

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 50
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 3000
    $CPU  = 1
    $MEMORY = 50
    $QUEUE = "long"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 3000
    $CPU  = 1
    $MEMORY = 50
    $QUEUE = "long"
}
//* platform
//* autofill

process Sequential_Mapping_Module_Download_build_sequential_mapping_indexes {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${commondbName}$/) "commondb/$filename"}
input:
 file genome from g245_54_genome10_g256_47
 file gtf from g245_54_gtfFile01_g256_47

output:
 file "${commondbName}"  into g256_47_commondb00_g256_43
 file "${bowtieIndex}" optional true  into g256_47_bowtieIndex11_g256_43
 file "${bowtie2Index}" optional true  into g256_47_bowtie2index22_g256_43
 file "${starIndex}" optional true  into g256_47_starIndex33_g256_43

when:
download_build_sequential_mapping_indexes == true && params.run_Sequential_Mapping == "yes"

script:
slashCount = params.commondb_source.count("/")
cutDir = slashCount - 3;
commondbSource = !file("${params.commondb}").exists() ? params.commondb_source : params.commondb
commondbName = "commondb"
inputName = file("${params.commondb}").getName().toString()

selectedSeqList = params.Sequential_Mapping_Module_Sequential_Mapping._select_sequence
alignerList = params.Sequential_Mapping_Module_Sequential_Mapping._aligner
genomeIndexes = selectedSeqList.findIndexValues { it ==~ /genome/ }
buildBowtieIndex = alignerList[genomeIndexes].contains("bowtie")
buildBowtie2Index = alignerList[genomeIndexes].contains("bowtie2")
buildSTARIndex = alignerList[genomeIndexes].contains("STAR")

basename = genome.baseName
bowtie2Index = "Bowtie2Index" 
bowtieIndex = "BowtieIndex" 
starIndex = "STARIndex"
bowtie_index_dir = ""
bowtie2_index_dir = ""
star_index_dir = ""
if (params.bowtie_index.indexOf('/') > -1 && params.bowtie_index.indexOf('s3://') < 0){
	bowtie_index_dir  = params.bowtie_index.substring(0, params.bowtie_index.lastIndexOf('/')) 
}
if (params.bowtie2_index.indexOf('/') > -1 && params.bowtie2_index.indexOf('s3://') < 0){
	bowtie2_index_dir  = params.bowtie2_index.substring(0, params.bowtie2_index.lastIndexOf('/')) 
}
if (params.star_index.indexOf('/') > -1 && params.star_index.indexOf('s3://') < 0){
	star_index_dir  = params.star_index.substring(0, params.star_index.lastIndexOf('/')) 
}

"""
if [ ! -e "${params.commondb}" ] ; then
    echo "${params.commondb} not found"
	if [[ "${params.commondb_source}" =~ "s3" ]]; then
		echo "Downloading s3 path from ${params.commondb_source}"
		aws s3 cp --recursive ${params.commondb_source} ${workDir}/${commondbName} && ln -s ${workDir}/${commondbName} ${commondbName}
	elif [[ "${params.commondb_source}" =~ "gs" ]]; then
		echo "Downloading gs path from ${params.commondb_source}"
		gsutil cp -r ${params.commondb_source} ${workDir}/. && ln -s ${workDir}/${commondbName} ${commondbName}
	else
		echo "Downloading commondb with wget"
		wget --no-check-certificate --secure-protocol=TLSv1 -l inf -nc -nH --cut-dirs=$cutDir -R 'index.html*' -r --no-parent --directory-prefix=\$PWD/${commondbName} ${params.commondb_source}
	fi

else 
	ln -s ${params.commondb} ${commondbName}
fi


if [ "${buildBowtie2Index}" == "true" ]; then
	if [ ! -e "${bowtie2_index_dir}/${basename}.rev.1.bt2" ] ; then
    	echo "${bowtie2_index_dir}/${basename}.rev.1.bt2 Bowtie2 index not found"
    	mkdir -p $bowtie2Index && cp $genome $gtf $bowtie2Index/. && cd $bowtie2Index
    	bowtie2-build ${genome} ${basename}
    	cd ..
    	if [ "${bowtie2_index_dir}" != "" ] ; then
			mkdir -p ${bowtie2_index_dir}
			cp -R -n $bowtie2Index  ${bowtie2_index_dir}
		fi
	else 
		ln -s ${bowtie2_index_dir} $bowtie2Index
	fi
fi

if [ "${buildBowtieIndex}" == "true" ]; then
	if [ ! -e "${bowtie_index_dir}/${basename}.rev.2.ebwt" ] ; then
    	echo "${bowtie_index_dir}/${basename}.rev.2.ebwt Bowtie index not found"
    	mkdir -p $bowtieIndex && cp $genome $gtf $bowtieIndex/. && cd $bowtieIndex
    	bowtie-build ${genome} ${basename}
    	cd ..
    	if [ "${bowtie_index_dir}" != "" ] ; then
			mkdir -p ${bowtie_index_dir}
			cp -R -n $bowtieIndex  ${bowtie_index_dir}
		fi
	else 
		ln -s ${bowtie_index_dir} $bowtieIndex
	fi
fi 

if [ "${buildSTARIndex}" == "true" ]; then
	if [ ! -e "${params.star_index}/SA" ] ; then
    	echo "STAR index not found"
    	mkdir -p $starIndex 
    	STAR --runMode genomeGenerate --genomeDir $starIndex --genomeFastaFiles ${genome} --sjdbGTFfile ${gtf}
		if [ "${star_index_dir}" != "" ] ; then
			mkdir -p ${star_index_dir}
			cp -R $starIndex  ${params.star_index}
		fi
	else 
		ln -s ${params.star_index} $starIndex
	fi
fi
"""




}

g256_47_commondb00_g256_43= g256_47_commondb00_g256_43.ifEmpty([""]) 
g256_47_bowtieIndex11_g256_43= g256_47_bowtieIndex11_g256_43.ifEmpty([""]) 
g256_47_bowtie2index22_g256_43= g256_47_bowtie2index22_g256_43.ifEmpty([""]) 
g256_47_starIndex33_g256_43= g256_47_starIndex33_g256_43.ifEmpty([""]) 

//* params.gtf =  ""  //* @input
//* params.genome =  ""  //* @input
//* params.commondb =  ""  //* @input
if (!(params.run_Sequential_Mapping  == "yes")){
g256_47_commondb00_g256_43.into{g256_43_commondb05_g256_44; g256_43_commondb05_g256_45; g256_43_commondb02_g256_46}
g256_47_bowtieIndex11_g256_43.into{g256_43_bowtieIndex12_g256_44; g256_43_bowtieIndex12_g256_45; g256_43_bowtieIndex13_g256_46}
g256_47_bowtie2index22_g256_43.into{g256_43_bowtie2index23_g256_44; g256_43_bowtie2index23_g256_45; g256_43_bowtie2index24_g256_46}
g256_47_starIndex33_g256_43.into{g256_43_starIndex34_g256_44; g256_43_starIndex34_g256_45; g256_43_starIndex35_g256_46}
} else {


process Sequential_Mapping_Module_Check_Sequential_Mapping_Indexes {

input:
 file commondb from g256_47_commondb00_g256_43
 file bowtieIndex from g256_47_bowtieIndex11_g256_43
 file bowtie2Index from g256_47_bowtie2index22_g256_43
 file starIndex from g256_47_starIndex33_g256_43

output:
 file "*/${commondb2}" optional true  into g256_43_commondb05_g256_44, g256_43_commondb05_g256_45, g256_43_commondb02_g256_46
 file "*/${bowtieIndex2}" optional true  into g256_43_bowtieIndex12_g256_44, g256_43_bowtieIndex12_g256_45, g256_43_bowtieIndex13_g256_46
 file "*/${bowtie2Index2}" optional true  into g256_43_bowtie2index23_g256_44, g256_43_bowtie2index23_g256_45, g256_43_bowtie2index24_g256_46
 file "*/${starIndex2}" optional true  into g256_43_starIndex34_g256_44, g256_43_starIndex34_g256_45, g256_43_starIndex35_g256_46

container 'quay.io/viascientific/pipeline_base_image:1.0'

when:
params.run_Sequential_Mapping  == "yes"

script:
(cmd1, commondb2) = pathChecker(commondb, params.commondb, "folder")
(cmd2, bowtieIndex2) = pathChecker(bowtieIndex, params.bowtie_index, "folder")
(cmd3, bowtie2Index2) = pathChecker(bowtie2Index, params.bowtie2_index, "folder")
(cmd4, starIndex2) = pathChecker(starIndex, params.star_index, "folder")
"""
$cmd1
$cmd2
$cmd3
$cmd4
"""
}
}


//* params.run_Adapter_Removal =   "no"   //* @dropdown @options:"yes","no" @show_settings:"Adapter_Removal"
//* @style @multicolumn:{seed_mismatches, palindrome_clip_threshold, simple_clip_threshold} @condition:{Tool_for_Adapter_Removal="trimmomatic", seed_mismatches, palindrome_clip_threshold, simple_clip_threshold}, {Tool_for_Adapter_Removal="fastx_clipper", discard_non_clipped}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 5
}
//* platform
//* platform
//* autofill
if (!((params.run_Adapter_Removal && (params.run_Adapter_Removal == "yes")) || !params.run_Adapter_Removal)){
g_347_reads00_g257_18.into{g257_18_reads01_g257_31; g257_18_reads00_g257_23}
g257_18_log_file10_g257_11 = Channel.empty()
} else {


process Adapter_Trimmer_Quality_Module_Adapter_Removal {

input:
 set val(name), file(reads) from g_347_reads00_g257_18
 val mate from g_347_mate11_g257_18

output:
 set val(name), file("reads/*.fastq.gz")  into g257_18_reads01_g257_31, g257_18_reads00_g257_23
 file "*.{fastx,trimmomatic}.log"  into g257_18_log_file10_g257_11

errorStrategy 'retry'

when:
(params.run_Adapter_Removal && (params.run_Adapter_Removal == "yes")) || !params.run_Adapter_Removal

shell:
phred = params.Adapter_Trimmer_Quality_Module_Adapter_Removal.phred
Tool_for_Adapter_Removal = params.Adapter_Trimmer_Quality_Module_Adapter_Removal.Tool_for_Adapter_Removal
Adapter_Sequence = params.Adapter_Trimmer_Quality_Module_Adapter_Removal.Adapter_Sequence
//trimmomatic_inputs
min_length = params.Adapter_Trimmer_Quality_Module_Adapter_Removal.min_length
seed_mismatches = params.Adapter_Trimmer_Quality_Module_Adapter_Removal.seed_mismatches
palindrome_clip_threshold = params.Adapter_Trimmer_Quality_Module_Adapter_Removal.palindrome_clip_threshold
simple_clip_threshold = params.Adapter_Trimmer_Quality_Module_Adapter_Removal.simple_clip_threshold

//fastx_clipper_inputs
discard_non_clipped = params.Adapter_Trimmer_Quality_Module_Adapter_Removal.discard_non_clipped
    
remove_previous_reads = params.Adapter_Trimmer_Quality_Module_Adapter_Removal.remove_previous_reads
discard_non_clipped_text = ""
if (discard_non_clipped == "yes") {discard_non_clipped_text = "-c"}
nameAll = reads.toString()
nameArray = nameAll.split(' ')
file2 = ""
if (nameAll.contains('.gz')) {
    newName =  nameArray[0] 
    file1 =  nameArray[0]
    if (mate == "pair") {file2 =  nameArray[1] }
} 
'''
#!/usr/bin/env perl
 use List::Util qw[min max];
 use strict;
 use File::Basename;
 use Getopt::Long;
 use Pod::Usage;
 use Cwd qw();
 
runCmd("mkdir reads adapter unpaired");

open(OUT, ">adapter/adapter.fa");
my @adaps=split(/\n/,"!{Adapter_Sequence}");
my $i=1;
foreach my $adap (@adaps)
{
 print OUT ">adapter$i\\n$adap\\n";
 $i++;
}
close(OUT);

my $quality="!{phred}";
print "fastq quality: $quality\\n";
print "tool: !{Tool_for_Adapter_Removal}\\n";

if ("!{mate}" eq "pair") {
    if ("!{Tool_for_Adapter_Removal}" eq "trimmomatic") {
        runCmd("trimmomatic PE -threads !{task.cpus} -phred${quality} !{file1} !{file2} reads/!{name}.1.fastq.gz unpaired/!{name}.1.fastq.unpaired.gz reads/!{name}.2.fastq.gz unpaired/!{name}.2.fastq.unpaired.gz ILLUMINACLIP:adapter/adapter.fa:!{seed_mismatches}:!{palindrome_clip_threshold}:!{simple_clip_threshold} MINLEN:!{min_length} 2> !{name}.trimmomatic.log");
    } elsif ("!{Tool_for_Adapter_Removal}" eq "fastx_clipper") {
        print "Fastx_clipper is not suitable for paired reads.";
    }
} else {
    if ("!{Tool_for_Adapter_Removal}" eq "trimmomatic") {
        runCmd("trimmomatic SE -threads !{task.cpus}  -phred${quality} !{file1} reads/!{name}.fastq.gz ILLUMINACLIP:adapter/adapter.fa:!{seed_mismatches}:!{palindrome_clip_threshold}:!{simple_clip_threshold} MINLEN:!{min_length} 2> !{name}.trimmomatic.log");
    } elsif ("!{Tool_for_Adapter_Removal}" eq "fastx_clipper") {
        runCmd("fastx_clipper  -Q $quality -a !{Adapter_Sequence} -l !{min_length} !{discard_non_clipped_text} -v -i !{file1} -o reads/!{name}.fastq.gz > !{name}.fastx.log");
    }
}
if ("!{remove_previous_reads}" eq "true") {
    my $currpath = Cwd::cwd();
    my @paths = (split '/', $currpath);
    splice(@paths, -2);
    my $workdir= join '/', @paths;
    splice(@paths, -1);
    my $inputsdir = join '/', @paths;
    $inputsdir .= "/work";
    print "INFO: inputs reads will be removed if they are located in the $workdir $inputsdir\\n";
    my @listOfFiles = `readlink -e !{file1} !{file2}`;
    foreach my $targetFile (@listOfFiles){
        if (index($targetFile, $workdir) != -1 || index($targetFile, $inputsdir) != -1) {
            runCmd("rm -f $targetFile");
            print "INFO: $targetFile deleted.\\n";
        }
    }
}


##Subroutines
sub runCmd {
    my ($com) = @_;
    if ($com eq ""){
		return "";
    }
    my $error = system(@_);
    if   ($error) { die "Command failed: $error $com\\n"; }
    else          { print "Command successful: $com\\n"; }
}
'''

}
}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 10
}
//* platform
//* platform
//* autofill

process Adapter_Trimmer_Quality_Module_FastQC_after_Adapter_Removal {

input:
 val mate from g_347_mate10_g257_31
 set val(name), file(reads) from g257_18_reads01_g257_31

output:
 file '*.{html,zip}'  into g257_31_FastQCout015_g_177

when:
(params.run_FastQC && params.run_FastQC == "yes" && params.run_Adapter_Removal && params.run_Adapter_Removal == "yes")

script:
"""
fastqc ${reads} 
"""
}


process Adapter_Trimmer_Quality_Module_Adapter_Removal_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /adapter_removal_detailed_summary.tsv$/) "adapter_removal_detailed_summary/$filename"}
input:
 file logfile from g257_18_log_file10_g257_11.collect()
 val mate from g_347_mate11_g257_11

output:
 file "adapter_removal_summary.tsv"  into g257_11_outputFileTSV05_g_198
 file "adapter_removal_detailed_summary.tsv" optional true  into g257_11_outputFile11

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;

my @header;
my %all_files;
my %tsv;
my %tsvDetail;
my %headerHash;
my %headerText;
my %headerTextDetail;

my $i = 0;
chomp( my $contents = `ls *.log` );

my @files = split( /[\\n]+/, $contents );
foreach my $file (@files) {
    $i++;
    my $mapOrder = "1";
    if ($file =~ /(.*)\\.fastx\\.log/){
        $file =~ /(.*)\\.fastx\\.log/;
        my $mapper   = "fastx";
        my $name = $1;    ##sample name
        push( @header, $mapper );

        my $in;
        my $out;
        my $tooshort;
        my $adapteronly;
        my $noncliped;
        my $Nreads;

        chomp( $in =`cat $file | grep 'Input:' | awk '{sum+=\\$2} END {print sum}'` );
        chomp( $out =`cat $file | grep 'Output:' | awk '{sum+=\\$2} END {print sum}'` );
        chomp( $tooshort =`cat $file | grep 'too-short reads' | awk '{sum+=\\$2} END {print sum}'`);
        chomp( $adapteronly =`cat $file | grep 'adapter-only reads' | awk '{sum+=\\$2} END {print sum}'`);
        chomp( $noncliped =`cat $file | grep 'non-clipped reads.' | awk '{sum+=\\$2} END {print sum}'`);
        chomp( $Nreads =`cat $file | grep 'N reads.' | awk '{sum+=\\$2} END {print sum}'` );

        $tsv{$name}{$mapper} = [ $in, $out ];
        $headerHash{$mapOrder} = $mapper;
        $headerText{$mapOrder} = [ "Total Reads", "Reads After Adapter Removal" ];
        $tsvDetail{$name}{$mapper} = [ $in, $tooshort, $adapteronly, $noncliped, $Nreads, $out ];
        $headerTextDetail{$mapOrder} = ["Total Reads","Too-short reads","Adapter-only reads","Non-clipped reads","N reads","Reads After Adapter Removal"];
    } elsif ($file =~ /(.*)\\.trimmomatic\\.log/){
        $file =~ /(.*)\\.trimmomatic\\.log/;
        my $mapper   = "trimmomatic";
        my $name = $1;    ##sample name
        push( @header, $mapper );
        
        my $in;
        my $out;

        if ( "!{mate}" eq "pair"){
            chomp( $in =`cat $file | grep 'Input Read Pairs:' | awk '{sum+=\\$4} END {print sum}'` );
            chomp( $out =`cat $file | grep 'Input Read Pairs:' | awk '{sum+=\\$7} END {print sum}'` );
        } else {
            chomp( $in =`cat $file | grep 'Input Reads:' | awk '{sum+=\\$3} END {print sum}'` );
            chomp( $out =`cat $file | grep 'Input Reads:' | awk '{sum+=\\$5} END {print sum}'` );
        }
        


        $tsv{$name}{$mapper} = [ $in, $out ];
        $headerHash{$mapOrder} = $mapper;
        $headerText{$mapOrder} = [ "Total Reads", "Reads After Adapter Removal" ];
        
    }
    
}

my @mapOrderArray = ( keys %headerHash );
my @sortedOrderArray = sort { $a <=> $b } @mapOrderArray;

my $summary          = "adapter_removal_summary.tsv";
my $detailed_summary = "adapter_removal_detailed_summary.tsv";
writeFile( $summary,          \\%headerText,       \\%tsv );
if (%headerTextDetail){
    writeFile( $detailed_summary, \\%headerTextDetail, \\%tsvDetail );  
}

sub writeFile {
    my $summary    = $_[0];
    my %headerText = %{ $_[1] };
    my %tsv        = %{ $_[2] };
    open( OUT, ">$summary" );
    print OUT "Sample\\t";
    my @headArr = ();
    for my $mapOrder (@sortedOrderArray) {
        push( @headArr, @{ $headerText{$mapOrder} } );
    }
    my $headArrAll = join( "\\t", @headArr );
    print OUT "$headArrAll\\n";

    foreach my $name ( keys %tsv ) {
        my @rowArr = ();
        for my $mapOrder (@sortedOrderArray) {
            push( @rowArr, @{ $tsv{$name}{ $headerHash{$mapOrder} } } );
        }
        my $rowArrAll = join( "\\t", @rowArr );
        print OUT "$name\\t$rowArrAll\\n";
    }
    close(OUT);
}

'''
}

//* @style @condition:{single_or_paired_end_reads="single", barcode_pattern1,remove_duplicates_based_on_UMI}, {single_or_paired_end_reads="pair", barcode_pattern1,barcode_pattern2}

if (!(params.run_UMIextract == "yes")){
g257_18_reads00_g257_23.set{g257_23_reads00_g257_19}
g257_23_log_file10_g257_24 = Channel.empty()
} else {


process Adapter_Trimmer_Quality_Module_UMIextract {

input:
 set val(name), file(reads) from g257_18_reads00_g257_23
 val mate from g_347_mate11_g257_23

output:
 set val(name), file("result/*.fastq.gz")  into g257_23_reads00_g257_19
 file "${name}.*.log"  into g257_23_log_file10_g257_24

container 'quay.io/viascientific/fastq_preprocessing:1.0'

when:
params.run_UMIextract == "yes" 

script:
readArray = reads.toString().split(' ')
file2 = ""
file1 =  readArray[0]
if (mate == "pair") {file2 =  readArray[1]}


single_or_paired_end_reads = params.Adapter_Trimmer_Quality_Module_UMIextract.single_or_paired_end_reads
barcode_pattern1 = params.Adapter_Trimmer_Quality_Module_UMIextract.barcode_pattern1
barcode_pattern2 = params.Adapter_Trimmer_Quality_Module_UMIextract.barcode_pattern2
UMIqualityFilterThreshold = params.Adapter_Trimmer_Quality_Module_UMIextract.UMIqualityFilterThreshold
phred = params.Adapter_Trimmer_Quality_Module_UMIextract.phred
remove_duplicates_based_on_UMI = params.Adapter_Trimmer_Quality_Module_UMIextract.remove_duplicates_based_on_UMI

"""
set +e
source activate umi_tools_env 2> /dev/null || true
mkdir result
if [ "${mate}" == "pair" ]; then
umi_tools extract --bc-pattern='${barcode_pattern1}' \
                  --bc-pattern2='${barcode_pattern2}' \
                  --extract-method=regex \
                  --stdin=${file1} \
                  --stdout=result/${name}_R1.fastq.gz \
                  --read2-in=${file2} \
                  --read2-out=result/${name}_R2.fastq.gz\
				  --quality-filter-threshold=${UMIqualityFilterThreshold} \
				  --quality-encoding=phred${phred} \
				  --log=${name}.umitools.log 


else
umi_tools extract --bc-pattern='${barcode_pattern1}' \
                  --log=${name}.umitools.log \
                  --extract-method=regex \
                  --stdin ${file1} \
                  --stdout result/${name}.fastq.gz \
				  --quality-filter-threshold=${UMIqualityFilterThreshold} \
				  --quality-encoding=phred${phred}
	if [ "${remove_duplicates_based_on_UMI}" == "true" ]; then		  
        mv result/${name}.fastq.gz  result/${name}_umitools.fastq.gz && gunzip result/${name}_umitools.fastq.gz
        ## only checks last part of the underscore splitted header for UMI
        awk '(NR%4==1){name=\$1;header=\$0;len=split(name,umiAr,"_");umi=umiAr[len];} (NR%4==2){total++;if(a[umi]!=1){nondup++;a[umi]=1;  print header;print;getline; print; getline; print;}} END{print FILENAME"\\t"total"\\t"nondup > "${name}.dedup.log"}' result/${name}_umitools.fastq > result/${name}.fastq
        rm result/${name}_umitools.fastq
        gzip result/${name}.fastq
	fi			  
fi
"""

}
}


//* params.run_Trimmer =   "no"   //* @dropdown @options:"yes","no" @show_settings:"Trimmer"
//* @style @multicolumn:{trim_length_5prime,trim_length_3prime}, {trim_length_5prime_R1,trim_length_3prime_R1}, {trim_length_5prime_R2,trim_length_3prime_R2} @condition:{single_or_paired_end_reads="single", trim_length_5prime,trim_length_3prime}, {single_or_paired_end_reads="pair", trim_length_5prime_R1,trim_length_3prime_R1,trim_length_5prime_R2,trim_length_3prime_R2}

//* autofill
//* platform
//* platform
//* autofill
if (!((params.run_Trimmer && (params.run_Trimmer == "yes")) || !params.run_Trimmer)){
g257_23_reads00_g257_19.set{g257_19_reads00_g257_20}
g257_19_log_file10_g257_21 = Channel.empty()
} else {


process Adapter_Trimmer_Quality_Module_Trimmer {

input:
 set val(name), file(reads) from g257_23_reads00_g257_19
 val mate from g_347_mate11_g257_19

output:
 set val(name), file("reads/*q.gz")  into g257_19_reads00_g257_20
 file "*.log" optional true  into g257_19_log_file10_g257_21

errorStrategy 'retry'

when:
(params.run_Trimmer && (params.run_Trimmer == "yes")) || !params.run_Trimmer

shell:
phred = params.Adapter_Trimmer_Quality_Module_Trimmer.phred
single_or_paired_end_reads = params.Adapter_Trimmer_Quality_Module_Trimmer.single_or_paired_end_reads
trim_length_5prime = params.Adapter_Trimmer_Quality_Module_Trimmer.trim_length_5prime
trim_length_3prime = params.Adapter_Trimmer_Quality_Module_Trimmer.trim_length_3prime
trim_length_5prime_R1 = params.Adapter_Trimmer_Quality_Module_Trimmer.trim_length_5prime_R1
trim_length_3prime_R1 = params.Adapter_Trimmer_Quality_Module_Trimmer.trim_length_3prime_R1
trim_length_5prime_R2 = params.Adapter_Trimmer_Quality_Module_Trimmer.trim_length_5prime_R2
trim_length_3prime_R2 = params.Adapter_Trimmer_Quality_Module_Trimmer.trim_length_3prime_R2
remove_previous_reads = params.Adapter_Trimmer_Quality_Module_Trimmer.remove_previous_reads



file1 =  reads[0] 
file2 = ""
if (mate == "pair") {file2 =  reads[1] }
rawFile1 = "_length_check1.fastq"
rawFile2 = "_length_check2.fastq"
'''
#!/usr/bin/env perl
 use List::Util qw[min max];
 use strict;
 use File::Basename;
 use Getopt::Long;
 use Pod::Usage; 
 use Cwd qw();
 
runCmd("mkdir reads");
runCmd("zcat !{file1} | head -n 100 > !{rawFile1}");
if ("!{mate}" eq "pair") {
	runCmd("zcat !{file2} | head -n 100 > !{rawFile2}");
}
my $file1 = "";
my $file2 = "";
if ("!{mate}" eq "pair") {
    $file1 = "!{file1}";
    $file2 = "!{file2}";
    my $trim1 = "!{trim_length_5prime_R1}:!{trim_length_3prime_R1}";
    my $trim2 = "!{trim_length_5prime_R2}:!{trim_length_3prime_R2}";
    my $len=getLength("!{rawFile1}");
    print "length of $file1: $len\\n";
    trimFiles($file1, $trim1, $len);
    my $len=getLength("!{rawFile2}");
    print "INFO: length of $file2: $len\\n";
    trimFiles($file2, $trim2, $len);
} else {
    $file1 = "!{file1}";
    my $trim1 = "!{trim_length_5prime}:!{trim_length_3prime}";
    my $len=getLength("!{rawFile1}");
    print "INFO: length of file1: $len\\n";
    trimFiles($file1, $trim1, $len);
}
if ("!{remove_previous_reads}" eq "true") {
    my $currpath = Cwd::cwd();
    my @paths = (split '/', $currpath);
    splice(@paths, -2);
    my $workdir= join '/', @paths;
    splice(@paths, -1);
    my $inputsdir= join '/', @paths;
    $inputsdir .= "/inputs";
    print "INFO: inputs reads will be removed if they are located in the workdir inputsdir\\n";
    my @listOfFiles = `readlink -e !{file1} !{file2}`;
    foreach my $targetFile (@listOfFiles){
        if (index($targetFile, $workdir) != -1 || index($targetFile, $inputsdir) != -1) {
            runCmd("rm -f $targetFile");
            print "INFO: $targetFile deleted.\\n";
        }
    }
}



sub trimFiles
{
  my ($file, $trim, $len)=@_;
    my @nts=split(/[,:\\s\\t]+/,$trim);
    my $inpfile="";
    my $com="";
    my $i=1;
    my $outfile="";
    my $param="";
    my $quality="-Q!{phred}";

    if (scalar(@nts)==2)
    {
      $param = "-f ".($nts[0]+1) if (exists($nts[0]) && $nts[0] >= 0 );
      $param .= " -l ".($len-$nts[1]) if (exists($nts[0]) && $nts[1] > 0 );
      $outfile="reads/$file";  
      $com="gunzip -c $file | fastx_trimmer $quality -v $param -z -o $outfile  > !{name}.fastx_trimmer.log" if ((exists($nts[0]) && $nts[0] > 0) || (exists($nts[0]) && $nts[1] > 0 ));
      print "INFO: $com\\n";
      if ($com eq ""){
          print "INFO: Trimmer skipped for $file \\n";
          runCmd("mv $file reads/.");
      } else {
          runCmd("$com");
          print "INFO: Trimmer executed for $file \\n";
      }
    }

    
}


sub getLength
{
   my ($filename)=@_;
   open (IN, $filename);
   my $j=1;
   my $len=0;
   while(my $line=<IN>)
   {
     chomp($line);
     if ($j >50) { last;}
     if ($j%4==0)
     {
        $len=length($line);
     }
     $j++;
   }
   close(IN);
   return $len;
}

sub runCmd {
    my ($com) = @_;
    if ($com eq ""){
		return "";
    }
    my $error = system(@_);
    if   ($error) { die "Command failed: $error $com\\n"; }
    else          { print "Command successful: $com\\n"; }
}

'''

}
}


//* params.run_Quality_Filtering =   "no"   //* @dropdown @options:"yes","no" @show_settings:"Quality_Filtering"
//* @style @multicolumn:{window_size,required_quality}, {leading,trailing,minlen}, {minQuality,minPercent} @condition:{tool="trimmomatic", minlen, trailing, leading, required_quality_for_window_trimming, window_size}, {tool="fastx", minQuality, minPercent}

//* autofill
//* platform
//* platform
//* autofill
if (!((params.run_Quality_Filtering && (params.run_Quality_Filtering == "yes")) || !params.run_Quality_Filtering)){
g257_19_reads00_g257_20.set{g257_20_reads00_g256_46}
g257_20_log_file10_g257_16 = Channel.empty()
} else {


process Adapter_Trimmer_Quality_Module_Quality_Filtering {

input:
 set val(name), file(reads) from g257_19_reads00_g257_20
 val mate from g_347_mate11_g257_20

output:
 set val(name), file("reads/*.gz")  into g257_20_reads00_g256_46
 file "*.{fastx,trimmomatic}_quality.log" optional true  into g257_20_log_file10_g257_16

when:
(params.run_Quality_Filtering && (params.run_Quality_Filtering == "yes")) || !params.run_Quality_Filtering    

shell:
tool = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.tool
phred = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.phred
window_size = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.window_size
required_quality_for_window_trimming = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.required_quality_for_window_trimming
leading = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.leading
trailing = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.trailing
minlen = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.minlen


// fastx parameters
minQuality = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.minQuality
minPercent = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.minPercent

remove_previous_reads = params.Adapter_Trimmer_Quality_Module_Quality_Filtering.remove_previous_reads

nameAll = reads.toString()
nameArray = nameAll.split(' ')
file2 ="";
if (nameAll.contains('.gz')) {
    file1 =  nameArray[0] 
    if (mate == "pair") {file2 =  nameArray[1]}
} 
'''
#!/usr/bin/env perl
 use List::Util qw[min max];
 use strict;
 use File::Basename;
 use Getopt::Long;
 use Pod::Usage; 
 use Cwd qw();
 
runCmd("mkdir reads unpaired");
my $param = "SLIDINGWINDOW:"."!{window_size}".":"."!{required_quality_for_window_trimming}";
$param.=" LEADING:"."!{leading}";
$param.=" TRAILING:"."!{trailing}";
$param.=" MINLEN:"."!{minlen}";

my $quality="!{phred}";

print "INFO: fastq quality: $quality\\n";
     
if ("!{tool}" eq "trimmomatic") {
    if ("!{mate}" eq "pair") {
        runCmd("trimmomatic PE -phred${quality} !{file1} !{file2} reads/!{name}.1.fastq.gz unpaired/!{name}.1.fastq.unpaired.gz reads/!{name}.2.fastq.gz unpaired/!{name}.1.fastq.unpaired.gz $param 2> !{name}.trimmomatic_quality.log");
    } else {
        runCmd("trimmomatic SE -phred${quality} !{file1} reads/!{name}.fastq.gz $param 2> !{name}.trimmomatic_quality.log");
    }
} elsif ("!{tool}" eq "fastx") {
    if ("!{mate}" eq "pair") {
        print("WARNING: Fastx option is not suitable for paired reads. This step will be skipped.");
        runCmd("mv !{file1} !{file2} reads/.");
    } else {
        runCmd("fastq_quality_filter  -Q $quality -q !{minQuality} -p !{minPercent} -v -i !{file1} -o reads/!{name}.fastq.gz > !{name}.fastx_quality.log");
    }
}
if ("!{remove_previous_reads}" eq "true") {
    my $currpath = Cwd::cwd();
    my @paths = (split '/', $currpath);
    splice(@paths, -2);
    my $workdir= join '/', @paths;
    splice(@paths, -1);
    my $inputsdir= join '/', @paths;
    $inputsdir .= "/inputs";
    print "INFO: inputs reads will be removed if they are located in the workdir inputsdir\\n";
    my @listOfFiles = `readlink -e !{file1} !{file2}`;
    foreach my $targetFile (@listOfFiles){
        if (index($targetFile, $workdir) != -1 || index($targetFile, $inputsdir) != -1) {
            runCmd("rm -f $targetFile");
            print "INFO: $targetFile deleted.\\n";
        }
    }
}

##Subroutines
sub runCmd {
    my ($com) = @_;
    if ($com eq ""){
		return "";
    }
    my $error = system(@_);
    if   ($error) { die "Command failed: $error $com\\n"; }
    else          { print "Command successful: $com\\n"; }
}


'''

}
}


g256_43_commondb02_g256_46= g256_43_commondb02_g256_46.ifEmpty([""]) 
g256_43_bowtieIndex13_g256_46= g256_43_bowtieIndex13_g256_46.ifEmpty([""]) 
g256_43_bowtie2index24_g256_46= g256_43_bowtie2index24_g256_46.ifEmpty([""]) 
g256_43_starIndex35_g256_46= g256_43_starIndex35_g256_46.ifEmpty([""]) 

//* params.bowtie_index =  ""  //* @input
//* params.bowtie2_index =  ""  //* @input
//* params.star_index =  ""  //* @input

//both bowtie and bowtie2 indexes located in same path
bowtieIndexes = [rRNA:  "commondb/rRNA/rRNA", ercc:  "commondb/ercc/ercc", miRNA: "commondb/miRNA/miRNA", tRNA:  "commondb/tRNA/tRNA", piRNA: "commondb/piRNA/piRNA", snRNA: "commondb/snRNA/snRNA", rmsk:  "commondb/rmsk/rmsk", mtRNA:  "commondb/mtRNA/mtRNA"]
genomeIndexes = [bowtie: "BowtieIndex", bowtie2: "Bowtie2Index", STAR: "STARIndex"]


//_nucleicAcidType="dna" should be defined in the autofill section of pipeline header in case dna is used.
_select_sequence = params.Sequential_Mapping_Module_Sequential_Mapping._select_sequence
index_directory = params.Sequential_Mapping_Module_Sequential_Mapping.index_directory
name_of_the_index_file = params.Sequential_Mapping_Module_Sequential_Mapping.name_of_the_index_file
_aligner = params.Sequential_Mapping_Module_Sequential_Mapping._aligner
aligner_Parameters = params.Sequential_Mapping_Module_Sequential_Mapping.aligner_Parameters
description = params.Sequential_Mapping_Module_Sequential_Mapping.description
filter_Out = params.Sequential_Mapping_Module_Sequential_Mapping.filter_Out
sense_antisense = params.Sequential_Mapping_Module_Sequential_Mapping.sense_antisense

desc_all=[]
description.eachWithIndex() {param,i -> 
    if (param.isEmpty()){
        desc_all[i] = name_of_the_index_file[i]
    }  else {
        desc_all[i] = param.replaceAll("[ |.|;]", "_")
    }
}
custom_index=[]
index_directory.eachWithIndex() {param,i -> 
    if (_select_sequence[i] == "genome"){
        custom_index[i] = genomeIndexes[_aligner[i]]
    }else if (_select_sequence[i] == "custom"){
        custom_index[i] = param+"/"+name_of_the_index_file[i]
    }else {
        custom_index[i] = bowtieIndexes[_select_sequence[i]]
    }
}

selectSequenceList = []
mapList = []
paramList = []
alignerList = []
filterList = []
indexList = []
senseList = []

//concat default mapping and custom mapping
mapList = (desc_all) 
paramList = (aligner_Parameters)
alignerList = (_aligner)
filterList = (filter_Out)
indexList = (custom_index)
senseList = (sense_antisense)
selectSequenceList = (_select_sequence)

mappingList = mapList.join(" ") // convert into space separated format in order to use in bash for loop
paramsList = paramList.join(",") // convert into comma separated format in order to use in as array in bash
alignersList = alignerList.join(",") 
filtersList = filterList.join(",") 
indexesList = indexList.join(",") 
senseList = senseList.join(",")
selectSequencesList = selectSequenceList.join(",")

//* @style @condition:{remove_duplicates="yes",remove_duplicates_based_on_UMI_after_mapping},{remove_duplicates="no"},{_select_sequence="custom", index_directory,name_of_the_index_file,description,_aligner,aligner_Parameters,filter_Out,sense_antisense},{_select_sequence=("rRNA","ercc","miRNA","tRNA","piRNA","snRNA","rmsk","mtRNA","genome"),_aligner,aligner_Parameters,filter_Out,sense_antisense}  @array:{_select_sequence,_select_sequence, index_directory,name_of_the_index_file,_aligner,aligner_Parameters,filter_Out,sense_antisense,description} @multicolumn:{_select_sequence,_select_sequence,index_directory,name_of_the_index_file,_aligner,aligner_Parameters,filter_Out, sense_antisense, description},{remove_duplicates,remove_duplicates_based_on_UMI_after_mapping}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 4
    $MEMORY = 20
}
//* platform
//* platform
//* autofill
if (!(params.run_Sequential_Mapping == "yes")){
g257_20_reads00_g256_46.into{g256_46_reads01_g249_14; g256_46_reads01_g248_36; g256_46_reads01_g268_44; g256_46_reads01_g264_31; g256_46_reads01_g250_26}
g256_46_bowfiles10_g256_26 = Channel.empty()
g256_46_bam_file20_g256_44 = Channel.empty()
g256_46_bam_file50_g256_45 = Channel.empty()
g256_46_bam_index31_g256_44 = Channel.empty()
g256_46_bam_index61_g256_45 = Channel.empty()
g256_46_filter42_g256_26 = Channel.empty()
g256_46_log_file70_g256_30 = Channel.empty()
g256_46_bigWig_file88 = Channel.empty()
} else {


process Sequential_Mapping_Module_Sequential_Mapping {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /bowfiles\/.?.*$/) "sequential_mapping/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*\/.*_sorted.bam$/) "sequential_mapping/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*\/.*_sorted.bam.bai$/) "sequential_mapping/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*\/.*_duplicates_stats.log$/) "sequential_mapping/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*\/.*.bw$/) "sequential_mapping/$filename"}
input:
 set val(name), file(reads) from g257_20_reads00_g256_46
 val mate from g_347_mate11_g256_46
 file commondb from g256_43_commondb02_g256_46
 file bowtie_index from g256_43_bowtieIndex13_g256_46
 file bowtie2_index from g256_43_bowtie2index24_g256_46
 file star_index from g256_43_starIndex35_g256_46

output:
 set val(name), file("final_reads/*.gz")  into g256_46_reads01_g249_14, g256_46_reads01_g248_36, g256_46_reads01_g268_44, g256_46_reads01_g264_31, g256_46_reads01_g250_26
 set val(name), file("bowfiles/?*") optional true  into g256_46_bowfiles10_g256_26
 file "*/*_sorted.bam" optional true  into g256_46_bam_file20_g256_44
 file "*/*_sorted.bam.bai" optional true  into g256_46_bam_index31_g256_44
 val filtersList  into g256_46_filter42_g256_26
 file "*/*_sorted.dedup.bam" optional true  into g256_46_bam_file50_g256_45
 file "*/*_sorted.dedup.bam.bai" optional true  into g256_46_bam_index61_g256_45
 file "*/*_duplicates_stats.log" optional true  into g256_46_log_file70_g256_30
 file "*/*.bw" optional true  into g256_46_bigWig_file88

when:
params.run_Sequential_Mapping == "yes"

script:
nameAll = reads.toString()
nameArray = nameAll.split(' ')
file2 = ""
file1 =  nameArray[0] 
if (mate == "pair") {file2 =  nameArray[1]}


remove_duplicates = params.Sequential_Mapping_Module_Sequential_Mapping.remove_duplicates
remove_duplicates_based_on_UMI_after_mapping = params.Sequential_Mapping_Module_Sequential_Mapping.remove_duplicates_based_on_UMI_after_mapping
create_bigWig = params.Sequential_Mapping_Module_Sequential_Mapping.create_bigWig
remove_previous_reads = params.Sequential_Mapping_Module_Sequential_Mapping.remove_previous_reads

"""
#!/bin/bash
mkdir reads final_reads bowfiles
workflowWorkDir=\$(cd ../../ && pwd)
if [ -n "${mappingList}" ]; then
    #rename files to standart format
    if [ "${mate}" == "pair" ]; then
        mv $file1 ${name}.1.fastq.gz 2>/dev/null
        mv $file2 ${name}.2.fastq.gz 2>/dev/null
        mv ${name}.1.fastq.gz ${name}.2.fastq.gz reads/.
    else
        mv $file1 ${name}.fastq.gz 2>/dev/null
        mv ${name}.fastq.gz reads/.
    fi
    #sequential mapping
    k=0
    prev="reads"
    IFS=',' read -r -a selectSeqListAr <<< "${selectSequencesList}"
    IFS=',' read -r -a paramsListAr <<< "${paramsList}" #create comma separated array 
    IFS=',' read -r -a filtersListAr <<< "${filtersList}"
    IFS=',' read -r -a indexesListAr <<< "${indexesList}"
    IFS=',' read -r -a alignersListAr <<< "${alignersList}"
    IFS=',' read -r -a senseListAr <<< "${senseList}"
    wrkDir=\$(pwd)
    startDir=\$(pwd)
    for rna_set in ${mappingList}
    do
        ((k++))
        printf -v k2 "%02d" "\$k" #turn into two digit format
        mkdir -p \${rna_set}/unmapped
        cd \$rna_set
        ## create link of the target file to prevent "too many symlinks error"
        for r in \${startDir}/\${prev}/*; do
            targetRead=\$(readlink -e \$r)
            rname=\$(basename \$r)
            echo "INFO: ln -s \$targetRead \$rname"
            ln -s \$targetRead \$rname
        done
        basename=""
        genomeDir="\${startDir}/\${indexesListAr[\$k-1]}"
        
        if [[ \${indexesListAr[\$k-1]} == s3* ]]; then
        	s3dir=\$(echo \${indexesListAr[\$k-1]} | sed 's|\\(.*\\)/.*|\\1|' | sed 's![^/]\$!&/!')
        	s3file=\$(echo \${indexesListAr[\$k-1]} | sed 's|.*\\/||')
        	# Remove fa/fasta suffix if it exists
        	s3file="\${s3file%.fa}"
        	s3file="\${s3file%.fasta}"
        	echo "INFO: s3dir: \$s3dir" 
        	echo "INFO: s3file: \$s3file" 
        	echo "INFO: rna_set: \$rna_set" 
        	aws s3 cp --recursive \${s3dir} \${startDir}/custom_seqs/\${rna_set}
        	genomeDir="\${startDir}/custom_seqs/\${rna_set}"
        	indexesListAr[\$k-1]="\${startDir}/custom_seqs/\${rna_set}/\${s3file}"
        fi
        
        if [ "\${selectSeqListAr[\$k-1]}" == "genome" ]; then
        	wrkDir="\${startDir}"
        	if [ "\${alignersListAr[\$k-1]}" == "bowtie" ]; then
        		basename="/\$(basename \${startDir}/\${indexesListAr[\$k-1]}/*.rev.1.ebwt | cut -d. -f1)"
        	elif [ "\${alignersListAr[\$k-1]}" == "bowtie2" ]; then
        		basename="/\$(basename \${startDir}/\${indexesListAr[\$k-1]}/*.rev.1.bt2 | cut -d. -f1)"
        	elif [ "\${alignersListAr[\$k-1]}" == "STAR" ]; then
        		basename="/\$(basename \${startDir}/\${indexesListAr[\$k-1]}/*.gtf | cut -d. -f1)"
        	fi
        elif [ "\${selectSeqListAr[\$k-1]}" == "custom" ] ; then
        	wrkDir=""
        fi
        echo "INFO: basename: \$basename"
        echo "INFO: genomeDir: \$genomeDir"
        echo "INFO: check bowtie index: \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.rev.1.ebwt"
        echo "INFO: check bowtie2 index: \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.1.bt2"
        echo "INFO: check star index: \${genomeDir}/SAindex"
        if [ -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.rev.1.ebwt" -o -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.1.bt2" -o  -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.fa"  -o  -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.fasta"  -o  -e "\${genomeDir}/SAindex" ]; then
            if [ -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.fa" ] ; then
                fasta=\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.fa
            elif [ -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.fasta" ] ; then
                fasta=\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.fasta
            fi
            echo "INFO: fasta: \$fasta"
            if [ -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.1.bt2" -a "\${alignersListAr[\$k-1]}" == "bowtie2" ] ; then
                echo "INFO: \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.1.bt2 Bowtie2 index found."
            elif [ -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.1.ebwt" -a "\${alignersListAr[\$k-1]}" == "bowtie" ] ; then
                echo "INFO: \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.1.ebwt Bowtie index found."
            elif [ -e "\$genomeDir/SAindex" -a "\${alignersListAr[\$k-1]}" == "STAR" ] ; then
                echo "INFO: \$genomeDir/SAindex STAR index found."
            elif [ -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.fa" -o  -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.fasta" ] ; then
                if [ "\${alignersListAr[\$k-1]}" == "bowtie2" ]; then
                    bowtie2-build \$fasta \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}
                elif [ "\${alignersListAr[\$k-1]}" == "STAR" ]; then
                    if [ -e "\${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.gtf" ]; then
                        STAR --runMode genomeGenerate --genomeDir \$genomeDir --genomeFastaFiles \$fasta --sjdbGTFfile \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.gtf --genomeSAindexNbases 5
                    else
                        echo "WARNING: \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}.gtf not found. STAR index is not generated."
                    fi
                elif [ "\${alignersListAr[\$k-1]}" == "bowtie" ]; then
                    bowtie-build \$fasta \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}
                fi
            fi
                
            if [ "${mate}" == "pair" ]; then
                if [ "\${alignersListAr[\$k-1]}" == "bowtie2" ]; then
                    bowtie2 \${paramsListAr[\$k-1]} -x \${wrkDir}/\${indexesListAr[\$k-1]}\${basename} --no-unal --un-conc-gz unmapped/${name}.unmapped.fastq.gz -1 ${name}.1.fastq.gz -2 ${name}.2.fastq.gz --al-conc-gz ${name}.fq.mapped.gz -S \${rna_set}_${name}_alignment.sam 2>&1 | tee \${k2}_${name}.bow_\${rna_set}
                    mv unmapped/${name}.unmapped.fastq.1.gz unmapped/${name}.unmapped.1.fastq.gz
                    mv unmapped/${name}.unmapped.fastq.2.gz unmapped/${name}.unmapped.2.fastq.gz
                elif [ "\${alignersListAr[\$k-1]}" == "STAR" ]; then
                    STAR \${paramsListAr[\$k-1]}  --genomeDir \$genomeDir --readFilesCommand zcat --readFilesIn ${name}.1.fastq.gz ${name}.2.fastq.gz --outSAMtype SAM  --outFileNamePrefix ${name}.star --outReadsUnmapped Fastx
                    mv ${name}.starAligned.out.sam \${rna_set}_${name}_alignment.sam
                    mv ${name}.starUnmapped.out.mate1 unmapped/${name}.unmapped.1.fastq
                    mv ${name}.starUnmapped.out.mate2 unmapped/${name}.unmapped.2.fastq
                    mv ${name}.starLog.final.out \${k2}_${name}.star_\${rna_set}
                    gzip unmapped/${name}.unmapped.1.fastq unmapped/${name}.unmapped.2.fastq
                elif [ "\${alignersListAr[\$k-1]}" == "bowtie" ]; then
                	gunzip -c ${name}.1.fastq.gz > ${name}.1.fastq
                	gunzip -c ${name}.2.fastq.gz > ${name}.2.fastq
                    bowtie \${paramsListAr[\$k-1]}   \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}  --un  unmapped/${name}.unmapped.fastq -1 ${name}.1.fastq -2 ${name}.2.fastq -S  \${rna_set}_${name}_alignment.sam 2>&1 | tee \${k2}_${name}.bow1_\${rna_set}  
                    rm ${name}.1.fastq ${name}.2.fastq
                    mv unmapped/${name}.unmapped_1.fastq unmapped/${name}.unmapped.1.fastq
                    mv unmapped/${name}.unmapped_2.fastq unmapped/${name}.unmapped.2.fastq
                    gzip unmapped/${name}.unmapped.1.fastq unmapped/${name}.unmapped.2.fastq
                fi
            else
                if [ "\${alignersListAr[\$k-1]}" == "bowtie2" ]; then
                    bowtie2 \${paramsListAr[\$k-1]} -x \${wrkDir}/\${indexesListAr[\$k-1]}\${basename} --no-unal --un-gz unmapped/${name}.unmapped.fastq -U ${name}.fastq.gz --al-gz ${name}.fq.mapped.gz -S \${rna_set}_${name}_alignment.sam 2>&1 | tee \${k2}_${name}.bow_\${rna_set}  
                elif [ "\${alignersListAr[\$k-1]}" == "STAR" ]; then
                    STAR \${paramsListAr[\$k-1]}  --genomeDir \$genomeDir --readFilesCommand zcat --readFilesIn ${name}.fastq.gz --outSAMtype SAM  --outFileNamePrefix ${name}.star --outReadsUnmapped Fastx
                    mv ${name}.starAligned.out.sam \${rna_set}_${name}_alignment.sam
                    mv ${name}.starUnmapped.out.mate1 unmapped/${name}.unmapped.fastq
                    mv ${name}.starLog.final.out \${k2}_${name}.star_\${rna_set}
                    gzip unmapped/${name}.unmapped.fastq
                elif [ "\${alignersListAr[\$k-1]}" == "bowtie" ]; then
                	gunzip -c ${name}.fastq.gz > ${name}.fastq
                    bowtie \${paramsListAr[\$k-1]}  \${wrkDir}/\${indexesListAr[\$k-1]}\${basename}  --un  unmapped/${name}.unmapped.fastq  ${name}.fastq  -S \${rna_set}_${name}_alignment.sam 2>&1 | tee \${k2}_${name}.bow1_\${rna_set}  
                    gzip unmapped/${name}.unmapped.fastq
                    rm  ${name}.fastq
                fi
            fi
            echo "INFO: samtools view -bT \${fasta} \${rna_set}_${name}_alignment.sam > \${rna_set}_${name}_alignment.bam"
            samtools view -bT \${fasta} \${rna_set}_${name}_alignment.sam > \${rna_set}_${name}_alignment.bam
            rm -f \${rna_set}_${name}_alignment.sam
            if [ "\${alignersListAr[\$k-1]}" == "bowtie" ]; then
                mv \${rna_set}_${name}_alignment.bam \${rna_set}_${name}_tmp0.bam
                echo "INFO: samtools view -F 0x04 -b \${rna_set}_${name}_tmp0.bam > \${rna_set}_${name}_alignment.bam"
                samtools view -F 0x04 -b \${rna_set}_${name}_tmp0.bam > \${rna_set}_${name}_alignment.bam  # Remove unmapped reads
                if [ "${mate}" == "pair" ]; then
                    echo "# unique mapped reads: \$(samtools view -f 0x40 -F 0x4 -q 255 \${rna_set}_${name}_alignment.bam | cut -f 1 | sort -T '.' | uniq | wc -l)" >> \${k2}_${name}.bow1_\${rna_set}
                else
                    echo "# unique mapped reads: \$(samtools view -F 0x40 -q 255 \${rna_set}_${name}_alignment.bam | cut -f 1 | sort -T '.' | uniq | wc -l)" >> \${k2}_${name}.bow1_\${rna_set}
                fi
            fi
            if [ "${mate}" == "pair" ]; then
                mv \${rna_set}_${name}_alignment.bam \${rna_set}_${name}_alignment.tmp1.bam
                echo "INFO: samtools sort -n -o \${rna_set}_${name}_alignment.tmp2 \${rna_set}_${name}_alignment.tmp1.bam"
                samtools sort -n -o \${rna_set}_${name}_alignment.tmp2.bam \${rna_set}_${name}_alignment.tmp1.bam 
                echo "INFO: samtools view -bf 0x02 \${rna_set}_${name}_alignment.tmp2.bam >\${rna_set}_${name}_alignment.bam"
                samtools view -bf 0x02 \${rna_set}_${name}_alignment.tmp2.bam >\${rna_set}_${name}_alignment.bam
                rm \${rna_set}_${name}_alignment.tmp1.bam \${rna_set}_${name}_alignment.tmp2.bam
            fi
            echo "INFO: samtools sort -o \${rna_set}@${name}_sorted.bam \${rna_set}_${name}_alignment.bam"
            samtools sort -o \${rna_set}@${name}_sorted.bam \${rna_set}_${name}_alignment.bam 
            echo "INFO: samtools index \${rna_set}@${name}_sorted.bam"
            samtools index \${rna_set}@${name}_sorted.bam
            
            if [ "${create_bigWig}" == "yes" ]; then
				echo "INFO: creating genome.sizes file"
				cat \$fasta | awk '\$0 ~ ">" {print c; c=0;printf substr(\$0,2,100) "\\t"; } \$0 !~ ">" {c+=length(\$0);} END { print c; }' > \${rna_set}.chrom.sizes && sed -i '1{/^\$/d}' \${rna_set}.chrom.sizes
				echo "INFO: creating bigWig file"
				bedtools genomecov -split -bg -ibam \${rna_set}@${name}_sorted.bam -g \${rna_set}.chrom.sizes > \${rna_set}@${name}.bg && wigToBigWig -clip -itemsPerSlot=1 \${rna_set}@${name}.bg \${rna_set}.chrom.sizes \${rna_set}@${name}.bw 
			fi
			
             # split sense and antisense bam files. 
            if [ "\${senseListAr[\$k-1]}" == "Yes" ]; then
                if [ "${mate}" == "pair" ]; then
                    echo "INFO: paired end sense antisense separation"
                	samtools view -f 65 -b \${rna_set}@${name}_sorted.bam >\${rna_set}@${name}_forward_sorted.bam
	                samtools index \${rna_set}@${name}_forward_sorted.bam
	                samtools view -F 16 -b \${rna_set}@${name}_forward_sorted.bam >\${rna_set}@${name}_sense_sorted.bam
	                samtools index \${rna_set}@${name}_sense_sorted.bam
	                samtools view -f 16 -b \${rna_set}@${name}_forward_sorted.bam >\${rna_set}@${name}_antisense_sorted.bam
	                samtools index \${rna_set}@${name}_antisense_sorted.bam
                else
	                echo "INFO: single end sense antisense separation"
	                samtools view -F 16 -b \${rna_set}@${name}_sorted.bam >\${rna_set}@${name}_sense_sorted.bam
	                samtools index \${rna_set}@${name}_sense_sorted.bam
	                samtools view -f 16 -b \${rna_set}@${name}_sorted.bam >\${rna_set}@${name}_antisense_sorted.bam
	                samtools index \${rna_set}@${name}_antisense_sorted.bam
                fi
                if [ "${create_bigWig}" == "yes" ]; then
					echo "INFO: creating bigWig file for sense antisense bam"
					bedtools genomecov -split -bg -ibam \${rna_set}@${name}_sense_sorted.bam -g \${rna_set}.chrom.sizes > \${rna_set}@${name}_sense.bg && wigToBigWig -clip -itemsPerSlot=1 \${rna_set}@${name}_sense.bg \${rna_set}.chrom.sizes \${rna_set}@${name}_sense.bw 
					bedtools genomecov -split -bg -ibam \${rna_set}@${name}_antisense_sorted.bam -g \${rna_set}.chrom.sizes > \${rna_set}@${name}_antisense.bg && wigToBigWig -clip -itemsPerSlot=1 \${rna_set}@${name}_antisense.bg \${rna_set}.chrom.sizes \${rna_set}@${name}_antisense.bw 
				fi
            fi
            
            
            
            if [ "${remove_duplicates}" == "yes" ]; then
                ## check read header whether they have UMI tags which are separated with underscore.(eg. NS5HGY:2:11_GTATAACCTT)
                umiCheck=\$(samtools view \${rna_set}@${name}_sorted.bam |head -n 1 | awk 'BEGIN {FS="\\t"}; {print \$1}' | awk 'BEGIN {FS=":"}; \$NF ~ /_/ {print \$NF}')
                
                # based on remove_duplicates_based_on_UMI_after_mapping
                if [ "${remove_duplicates_based_on_UMI_after_mapping}" == "yes" -a ! -z "\$umiCheck" ]; then
                    echo "INFO: umi_mark_duplicates.py will be executed for removing duplicates from bam file"
                    echo "python umi_mark_duplicates.py -f \${rna_set}@${name}_sorted.bam -p 4"
                    python umi_mark_duplicates.py -f \${rna_set}@${name}_sorted.bam -p 4
                else
                    echo "INFO: Picard MarkDuplicates will be executed for removing duplicates from bam file"
                    if [ "${remove_duplicates_based_on_UMI_after_mapping}" == "yes"  ]; then
                        echo "WARNING: Read header have no UMI tags which are separated with underscore. Picard MarkDuplicates will be executed to remove duplicates from alignment file (bam) instead of remove_duplicates_based_on_UMI_after_mapping."
                    fi
                    echo "INFO: picard MarkDuplicates OUTPUT=\${rna_set}@${name}_sorted.deumi.sorted.bam METRICS_FILE=${name}_picard_PCR_duplicates.log  VALIDATION_STRINGENCY=LENIENT REMOVE_DUPLICATES=false INPUT=\${rna_set}@${name}_sorted.bam"
                    picard MarkDuplicates OUTPUT=\${rna_set}@${name}_sorted.deumi.sorted.bam METRICS_FILE=${name}_picard_PCR_duplicates.log  VALIDATION_STRINGENCY=LENIENT REMOVE_DUPLICATES=false INPUT=\${rna_set}@${name}_sorted.bam 
                fi
                #get duplicates stats (read the sam flags)
                samtools flagstat \${rna_set}@${name}_sorted.deumi.sorted.bam > \${k2}@\${rna_set}@${name}_duplicates_stats.log
                #remove alignments marked as duplicates
                samtools view -b -F 0x400 \${rna_set}@${name}_sorted.deumi.sorted.bam > \${rna_set}@${name}_sorted.deumi.sorted.bam.x_dup
                #sort deduplicated files by chrom pos
                echo "INFO: samtools sort -o \${rna_set}@${name}_sorted.dedup.bam \${rna_set}@${name}_sorted.deumi.sorted.bam.x_dup"
                samtools sort -o \${rna_set}@${name}_sorted.dedup.bam \${rna_set}@${name}_sorted.deumi.sorted.bam.x_dup 
                samtools index \${rna_set}@${name}_sorted.dedup.bam
                #get flagstat after dedup
                echo "##After Deduplication##" >> \${k2}@\${rna_set}@${name}_duplicates_stats.log
                samtools flagstat \${rna_set}@${name}_sorted.dedup.bam >> \${k2}@\${rna_set}@${name}_duplicates_stats.log
                if [ "${create_bigWig}" == "yes" ]; then
					echo "INFO: creating bigWig file for dedup bam"
					bedtools genomecov -split -bg -ibam \${rna_set}@${name}_sorted.dedup.bam -g \${rna_set}.chrom.sizes > \${rna_set}@${name}_dedup.bg && wigToBigWig -clip -itemsPerSlot=1 \${rna_set}@${name}_dedup.bg \${rna_set}.chrom.sizes \${rna_set}@${name}_dedup.bw 
				fi
                
                # split sense and antisense bam files. 
	            if [ "\${senseListAr[\$k-1]}" == "Yes" ]; then
	                if [ "${mate}" == "pair" ]; then
	                    echo "INFO: paired end sense antisense separation"
	                	samtools view -f 65 -b \${rna_set}@${name}_sorted.dedup.bam >\${rna_set}@${name}_forward_sorted.dedup.bam
		                samtools index \${rna_set}@${name}_forward_sorted.dedup.bam
		                samtools view -F 16 -b \${rna_set}@${name}_forward_sorted.dedup.bam>\${rna_set}@${name}_sense_sorted.dedup.bam
		                samtools index \${rna_set}@${name}_sense_sorted.dedup.bam
		                samtools view -f 16 -b \${rna_set}@${name}_forward_sorted.dedup.bam >\${rna_set}@${name}_antisense_sorted.dedup.bam
		                samtools index \${rna_set}@${name}_antisense_sorted.dedup.bam
	                else
		                echo "INFO: single end sense antisense separation"
		                samtools view -F 16 -b \${rna_set}@${name}_sorted.dedup.bam >\${rna_set}@${name}_sense_sorted.dedup.bam
		                samtools index \${rna_set}@${name}_sense_sorted.dedup.bam
		                samtools view -f 16 -b \${rna_set}@${name}_sorted.dedup.bam >\${rna_set}@${name}_antisense_sorted.dedup.bam
		                samtools index \${rna_set}@${name}_antisense_sorted.dedup.bam
	                fi
	                if [ "${create_bigWig}" == "yes" ]; then
						echo "INFO: creating bigWig file for sense antisense bam"
						bedtools genomecov -split -bg -ibam \${rna_set}@${name}_sense_sorted.dedup.bam -g \${rna_set}.chrom.sizes > \${rna_set}@${name}_sense_sorted.dedup.bg && wigToBigWig -clip -itemsPerSlot=1 \${rna_set}@${name}_sense_sorted.dedup.bg \${rna_set}.chrom.sizes \${rna_set}@${name}_sense_sorted.dedup.bw
						bedtools genomecov -split -bg -ibam \${rna_set}@${name}_antisense_sorted.dedup.bam -g \${rna_set}.chrom.sizes > \${rna_set}@${name}_antisense_sorted.dedup.bg && wigToBigWig -clip -itemsPerSlot=1 \${rna_set}@${name}_antisense_sorted.dedup.bg \${rna_set}.chrom.sizes \${rna_set}@${name}_antisense_sorted.dedup.bw
					fi
	            fi
            fi
            
        
            for file in unmapped/*; do mv \$file \${file/.unmapped/}; done ##remove .unmapped from filename
            if [ "\${alignersListAr[\$k-1]}" == "bowtie2" ]; then
                grep -v Warning \${k2}_${name}.bow_\${rna_set} > ${name}.tmp
                mv ${name}.tmp \${k2}_${name}.bow_\${rna_set}
                cp \${k2}_${name}.bow_\${rna_set} ./../bowfiles/.
            elif [ "\${alignersListAr[\$k-1]}" == "bowtie" ]; then
                cp \${k2}_${name}.bow1_\${rna_set} ./../bowfiles/.
            elif [ "\${alignersListAr[\$k-1]}" == "STAR" ]; then
                cp \${k2}_${name}.star_\${rna_set} ./../bowfiles/.
            fi
            cd ..
            # if filter is on, remove previously created unmapped fastq. 
            if [ "\${filtersListAr[\$k-1]}" == "Yes" ]; then
                if [ "\${prev}" != "reads" ]; then
                    echo "INFO: remove prev: \${prev}/*"
                    rm -rf \${prev}/*
                elif  [ "${remove_previous_reads}" == "true" ]; then
                    echo "INFO: inputs reads will be removed if they are located in the workdir"
                    for f in \${prev}/*; do
                        targetFile=\$(readlink -e \$f)
                        echo "INFO: targetFile: \$targetFile"
                        if [[ \$targetFile == *"\${workflowWorkDir}"* ]]; then
                            rm -f \$targetFile
                            echo "INFO: \$targetFile located in workdir and deleted."
                        fi
                    done
                fi
            # if filter is off remove current unmapped fastq
            else
                echo "INFO: remove \${rna_set}/unmapped/*"
                rm -rf \${rna_set}/unmapped/*
            fi
        else
            echo "WARNING: \${startDir}/\${indexesListAr[\$k-1]}\${basename} Mapping skipped. File not found."
            cd unmapped 
            ln -s \${startDir}/\${rna_set}/*fastq.gz .
            cd ..
            cd ..
        fi
        
        if [ "\${filtersListAr[\$k-1]}" == "Yes" ]; then
            prev=\${rna_set}/unmapped
        fi
    done
    cd final_reads && ln -s \${startDir}/\${prev}/* .
else 
    mv ${reads} final_reads/.
fi
## fix for google cloud, it cannot publish symlinks which has reference to outside of the workdir
for file in "\${startDir}/final_reads"/*; do
	echo "INFO: file in final_reads folder: \$file" 
    if [ -h "\$file" ]; then
        # Check if it's a symlink
        link_target=\$(readlink "\$file")
        if [ -e "\$link_target" ]; then
            # If the target of the symlink exists, replace the symlink with the target
            rm "\$file"
            cp -a "\$link_target" "\$file"
            echo "INFO: Replaced symlink '\$file' with the original file."
        else
            # If the target doesn't exist, the symlink is broken
            echo "INFO: Broken symlink: '\$file'"
        fi
    fi
done
"""

}
}


g256_43_bowtieIndex12_g256_45= g256_43_bowtieIndex12_g256_45.ifEmpty([""]) 
g256_43_bowtie2index23_g256_45= g256_43_bowtie2index23_g256_45.ifEmpty([""]) 
g256_43_starIndex34_g256_45= g256_43_starIndex34_g256_45.ifEmpty([""]) 


process Sequential_Mapping_Module_Sequential_Mapping_Bam_Dedup_count {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.counts.tsv$/) "Sequential_Mapping_Bam_dedup_count/$filename"}
input:
 file bam from g256_46_bam_file50_g256_45.collect()
 file index from g256_46_bam_index61_g256_45.collect()
 file bowtie_index from g256_43_bowtieIndex12_g256_45
 file bowtie2_index from g256_43_bowtie2index23_g256_45
 file star_index from g256_43_starIndex34_g256_45
 file commondb from g256_43_commondb05_g256_45

output:
 file "*.counts.tsv"  into g256_45_outputFileTSV00

shell:
mappingListQuoteSep = mapList.collect{ '"' + it + '"'}.join(",")
rawIndexList = indexList.collect{ '"' + it + '"'}.join(",")
selectSeqListQuote = selectSequenceList.collect{ '"' + it + '"'}.join(",")
alignerListQuote = alignerList.collect{ '"' + it + '"'}.join(",")
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;

my @header;
my @header_antisense;
my @header_sense;
my %all_files;
my %sense_files;
my %antisense_files;

my @mappingList = (!{mappingListQuoteSep});
my @rawIndexList = (!{rawIndexList});
my @selectSeqList = (!{selectSeqListQuote});
my @alignerList = (!{alignerListQuote});

my %indexHash;
my %selectSeqHash;
my %alignerHash;
my $dedup = "";
@indexHash{@mappingList} = @rawIndexList;
@selectSeqHash{@mappingList} = @selectSeqList;
@alignerHash{@mappingList} = @alignerList;

chomp(my $contents = `ls *.bam`);
my @files = split(/[\\n]+/, $contents);
foreach my $file (@files){
        $file=~/(.*)@(.*)_sorted(.*)\\.bam/;
        my $mapper = $1; 
        my $name = $2; ##header
        #print $3;
        if ($3 eq ".dedup"){
            $dedup = ".dedup";
        }
        if ($name=~/_antisense$/){
        	push(@header_antisense, $name) unless grep{$_ eq $name} @header_antisense; #mapped element header
        	$antisense_files{$mapper} .= $file." ";
        }
        elsif ($name=~/_sense$/){
        	push(@header_sense, $name) unless grep{$_ eq $name} @header_sense; #mapped element header
        	$sense_files{$mapper} .= $file." ";
        }
        else{
			push(@header, $name) unless grep{$_ eq $name} @header; #mapped element header
	        $all_files{$mapper} .= $file." ";
        }
}

runCov(\\%all_files, \\@header, \\@indexHash, "", $dedup);
runCov(\\%sense_files, \\@header_sense, \\@indexHash, "sense", $dedup);
runCov(\\%antisense_files, \\@header_antisense, \\@indexHash, "antisense", $dedup);

sub runCov {
	my ( \$files, \$header, \$indexHash, \$sense_antisense, \$dedup) = @_;
	open OUT, ">header".\$sense_antisense.".tsv";
	print OUT join ("\\t", "id","len",@{\$header}),"\\n";
	close OUT;
	my $par = "";
	if ($sense_antisense=~/^sense\$/){
      $par = "-s";
    }elsif($sense_antisense=~/^antisense\$/){
      $par = "-S";
    }
	
	foreach my $key (sort keys %{\$files}) {  
	   my $bamFiles = ${\$files}{$key};
	   
	   my $prefix = ${indexHash}{$key};
	   my $selectedSeq = ${selectSeqHash}{$key};
	   my $aligner = ${alignerHash}{$key};
	   if ($selectedSeq eq "genome"){
	   	  if ($aligner eq "bowtie"){
	   		$basename = `basename $prefix/*.rev.1.ebwt | cut -d. -f1`;
	   	  } elsif ($aligner eq "bowtie2"){
	   		$basename = `basename $prefix/*.rev.1.bt2 | cut -d. -f1`;
	   	  } elsif ($aligner eq "STAR"){
	   	    $basename = `basename $prefix/*.gtf | cut -d. -f1`;
	   	  }
	   	  $basename =~ s|\\s*$||;
	   	  $prefix = $prefix."/".$basename;
	   }  elsif($selectedSeq eq "custom"){
	   	  if (substr($prefix, 0, length("s3")) eq "s3"){
	   	  	 my $filename = (split '/', $prefix)[-1];
	   	  	 $filename =~ s/\\.fa//;
	   	  	 $prefix = "custom_seqs/${key}/${filename}";
	   	  	 print "new prefix $prefix\\n";
	   	  }
       }
	   
		unless (-e $prefix.".bed") {
            print "2: bed not found run makeBed\\n";
                if (-e $prefix.".fa") {
                    makeBed($prefix.".fa", $key, $prefix.".bed");
                } elsif(-e $prefix.".fasta"){
                    makeBed($prefix.".fasta", $key, $prefix.".bed");
                }
        }
	    
		my $com =  "bedtools multicov $par -bams $bamFiles -bed ".$prefix.".bed > $key${dedup}${sense_antisense}.counts.tmp\\n";
        print $com;
        `$com`;
        my $iniResColumn = int(countColumn($prefix.".bed")) + 1;
	    `awk -F \\"\\\\t\\" \\'{a=\\"\\";for (i=$iniResColumn;i<=NF;i++){a=a\\"\\\\t\\"\\$i;} print \\$4\\"\\\\t\\"(\\$3-\\$2)\\"\\"a}\\' $key${dedup}${sense_antisense}.counts.tmp> $key${dedup}${sense_antisense}.counts.tsv`;
	    `sort -k3,3nr $key${dedup}${sense_antisense}.counts.tsv>$key${dedup}${sense_antisense}.sorted.tsv`;
        `cat header${sense_antisense}.tsv $key${dedup}${sense_antisense}.sorted.tsv> $key${dedup}${sense_antisense}.counts.tsv`;
	}
}

sub countColumn {
    my ( \$file) = @_;
    open(IN, \$file);
    my $line=<IN>;
    chomp($line);
    my @cols = split('\\t', $line);
    my $n = @cols;
    close OUT;
    return $n;
}

sub makeBed {
    my ( \$fasta, \$type, \$bed) = @_;
    print "makeBed $fasta\\n";
    print "makeBed $bed\\n";
    open OUT, ">$bed";
    open(IN, \$fasta);
    my $name="";
    my $seq="";
    my $i=0;
    while(my $line=<IN>){
        chomp($line);
        if($line=~/^>(.*)/){
            $i++ if (length($seq)>0);
            print OUT "$name\\t1\\t".length($seq)."\\t$name\\t0\\t+\\n" if (length($seq)>0); 
            $name="$1";
            $seq="";
        } elsif($line=~/[ACGTNacgtn]+/){
            $seq.=$line;
        }
    }
    $name=~s/\\r//g;
    print OUT "$name\\t1\\t".length($seq)."\\t$name\\t0\\t+\\n" if (length($seq)>0); 
    close OUT;
}

'''
}

g256_43_bowtieIndex12_g256_44= g256_43_bowtieIndex12_g256_44.ifEmpty([""]) 
g256_43_bowtie2index23_g256_44= g256_43_bowtie2index23_g256_44.ifEmpty([""]) 
g256_43_starIndex34_g256_44= g256_43_starIndex34_g256_44.ifEmpty([""]) 


process Sequential_Mapping_Module_Sequential_Mapping_Bam_count {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.counts.tsv$/) "Sequential_Mapping_Bam_count/$filename"}
input:
 file bam from g256_46_bam_file20_g256_44.collect()
 file index from g256_46_bam_index31_g256_44.collect()
 file bowtie_index from g256_43_bowtieIndex12_g256_44
 file bowtie2_index from g256_43_bowtie2index23_g256_44
 file star_index from g256_43_starIndex34_g256_44
 file commondb from g256_43_commondb05_g256_44

output:
 file "*.counts.tsv"  into g256_44_outputFileTSV00

shell:
mappingListQuoteSep = mapList.collect{ '"' + it + '"'}.join(",")
rawIndexList = indexList.collect{ '"' + it + '"'}.join(",")
selectSeqListQuote = selectSequenceList.collect{ '"' + it + '"'}.join(",")
alignerListQuote = alignerList.collect{ '"' + it + '"'}.join(",")
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;

my @header;
my @header_antisense;
my @header_sense;
my %all_files;
my %sense_files;
my %antisense_files;

my @mappingList = (!{mappingListQuoteSep});
my @rawIndexList = (!{rawIndexList});
my @selectSeqList = (!{selectSeqListQuote});
my @alignerList = (!{alignerListQuote});

my %indexHash;
my %selectSeqHash;
my %alignerHash;
my $dedup = "";
@indexHash{@mappingList} = @rawIndexList;
@selectSeqHash{@mappingList} = @selectSeqList;
@alignerHash{@mappingList} = @alignerList;

chomp(my $contents = `ls *.bam`);
my @files = split(/[\\n]+/, $contents);
foreach my $file (@files){
        $file=~/(.*)@(.*)_sorted(.*)\\.bam/;
        my $mapper = $1; 
        my $name = $2; ##header
        #print $3;
        if ($3 eq ".dedup"){
            $dedup = ".dedup";
        }
        if ($name=~/_antisense$/){
        	push(@header_antisense, $name) unless grep{$_ eq $name} @header_antisense; #mapped element header
        	$antisense_files{$mapper} .= $file." ";
        }
        elsif ($name=~/_sense$/){
        	push(@header_sense, $name) unless grep{$_ eq $name} @header_sense; #mapped element header
        	$sense_files{$mapper} .= $file." ";
        }
        else{
			push(@header, $name) unless grep{$_ eq $name} @header; #mapped element header
	        $all_files{$mapper} .= $file." ";
        }
}

runCov(\\%all_files, \\@header, \\@indexHash, "", $dedup);
runCov(\\%sense_files, \\@header_sense, \\@indexHash, "sense", $dedup);
runCov(\\%antisense_files, \\@header_antisense, \\@indexHash, "antisense", $dedup);

sub runCov {
	my ( \$files, \$header, \$indexHash, \$sense_antisense, \$dedup) = @_;
	open OUT, ">header".\$sense_antisense.".tsv";
	print OUT join ("\\t", "id","len",@{\$header}),"\\n";
	close OUT;
	my $par = "";
	if ($sense_antisense=~/^sense\$/){
      $par = "-s";
    }elsif($sense_antisense=~/^antisense\$/){
      $par = "-S";
    }
	
	foreach my $key (sort keys %{\$files}) {  
	   my $bamFiles = ${\$files}{$key};
	   
	   my $prefix = ${indexHash}{$key};
	   my $selectedSeq = ${selectSeqHash}{$key};
	   my $aligner = ${alignerHash}{$key};
	   if ($selectedSeq eq "genome"){
	   	  if ($aligner eq "bowtie"){
	   		$basename = `basename $prefix/*.rev.1.ebwt | cut -d. -f1`;
	   	  } elsif ($aligner eq "bowtie2"){
	   		$basename = `basename $prefix/*.rev.1.bt2 | cut -d. -f1`;
	   	  } elsif ($aligner eq "STAR"){
	   	    $basename = `basename $prefix/*.gtf | cut -d. -f1`;
	   	  }
	   	  $basename =~ s|\\s*$||;
	   	  $prefix = $prefix."/".$basename;
	   }  elsif($selectedSeq eq "custom"){
	   	  if (substr($prefix, 0, length("s3")) eq "s3"){
	   	  	 my $filename = (split '/', $prefix)[-1];
	   	  	 $filename =~ s/\\.fa//;
	   	  	 $prefix = "custom_seqs/${key}/${filename}";
	   	  	 print "new prefix $prefix\\n";
	   	  }
       }
	   
		unless (-e $prefix.".bed") {
            print "2: bed not found run makeBed\\n";
                if (-e $prefix.".fa") {
                    makeBed($prefix.".fa", $key, $prefix.".bed");
                } elsif(-e $prefix.".fasta"){
                    makeBed($prefix.".fasta", $key, $prefix.".bed");
                }
        }
	    
		my $com =  "bedtools multicov $par -bams $bamFiles -bed ".$prefix.".bed > $key${dedup}${sense_antisense}.counts.tmp\\n";
        print $com;
        `$com`;
        my $iniResColumn = int(countColumn($prefix.".bed")) + 1;
	    `awk -F \\"\\\\t\\" \\'{a=\\"\\";for (i=$iniResColumn;i<=NF;i++){a=a\\"\\\\t\\"\\$i;} print \\$4\\"\\\\t\\"(\\$3-\\$2)\\"\\"a}\\' $key${dedup}${sense_antisense}.counts.tmp> $key${dedup}${sense_antisense}.counts.tsv`;
	    `sort -k3,3nr $key${dedup}${sense_antisense}.counts.tsv>$key${dedup}${sense_antisense}.sorted.tsv`;
        `cat header${sense_antisense}.tsv $key${dedup}${sense_antisense}.sorted.tsv> $key${dedup}${sense_antisense}.counts.tsv`;
	}
}

sub countColumn {
    my ( \$file) = @_;
    open(IN, \$file);
    my $line=<IN>;
    chomp($line);
    my @cols = split('\\t', $line);
    my $n = @cols;
    close OUT;
    return $n;
}

sub makeBed {
    my ( \$fasta, \$type, \$bed) = @_;
    print "makeBed $fasta\\n";
    print "makeBed $bed\\n";
    open OUT, ">$bed";
    open(IN, \$fasta);
    my $name="";
    my $seq="";
    my $i=0;
    while(my $line=<IN>){
        chomp($line);
        if($line=~/^>(.*)/){
            $i++ if (length($seq)>0);
            print OUT "$name\\t1\\t".length($seq)."\\t$name\\t0\\t+\\n" if (length($seq)>0); 
            $name="$1";
            $seq="";
        } elsif($line=~/[ACGTNacgtn]+/){
            $seq.=$line;
        }
    }
    $name=~s/\\r//g;
    print OUT "$name\\t1\\t".length($seq)."\\t$name\\t0\\t+\\n" if (length($seq)>0); 
    close OUT;
}

'''
}


process Sequential_Mapping_Module_Deduplication_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /deduplication_summary.tsv$/) "sequential_mapping_summary/$filename"}
input:
 file flagstat from g256_46_log_file70_g256_30.collect()
 val mate from g_347_mate11_g256_30

output:
 file "deduplication_summary.tsv"  into g256_30_outputFileTSV00

errorStrategy 'retry'
maxRetries 2

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;

my @header;
my %all_files;
my %tsv;
my %headerHash;
my %headerText;

my $i=0;
chomp(my $contents = `ls *_duplicates_stats.log`);
my @files = split(/[\\n]+/, $contents);
foreach my $file (@files){
    $i++;
    $file=~/(.*)@(.*)@(.*)_duplicates_stats\\.log/;
    my $mapOrder = int($1); 
    my $mapper = $2; #mapped element 
    my $name = $3; ##sample name
    push(@header, $mapper) unless grep{$_ eq $mapper} @header; 
        
    # my $duplicates;
    my $aligned;
    my $dedup; #aligned reads after dedup
    my $percent=0;
    if ("!{mate}" eq "pair" ){
        #first flagstat belongs to first bam file
        chomp($aligned = `cat $file | grep 'properly paired (' | sed -n 1p | awk '{sum+=\\$1+\\$3} END {print sum}'`);
        #second flagstat belongs to dedup bam file
        chomp($dedup = `cat $file | grep 'properly paired (' | sed -n 2p | awk '{sum+=\\$1+\\$3} END {print sum}'`);
    } else {
        chomp($aligned = `cat $file | grep 'mapped (' | sed -n 1p | awk '{sum+=\\$1+\\$3} END {print sum}'`);
        chomp($dedup = `cat $file | grep 'mapped (' | sed -n 2p | awk '{sum+=\\$1+\\$3} END {print sum}'`);
    }
    # chomp($duplicates = `cat $file | grep 'duplicates' | awk '{sum+=\\$1+\\$3} END {print sum}'`);
    # $dedup = int($aligned) - int($duplicates);
    if ("!{mate}" eq "pair" ){
       $dedup = int($dedup/2);
       $aligned = int($aligned/2);
    } 
    $percent = "0.00";
    if (int($aligned)  > 0 ){
       $percent = sprintf("%.2f", ($aligned-$dedup)/$aligned*100); 
    } 
    $tsv{$name}{$mapper}=[$aligned,$dedup,"$percent%"];
    $headerHash{$mapOrder}=$mapper;
    $headerText{$mapOrder}=["$mapper (Before Dedup)", "$mapper (After Dedup)", "$mapper (Duplication Ratio %)"];
}

my @mapOrderArray = ( keys %headerHash );
my @sortedOrderArray = sort { $a <=> $b } @mapOrderArray;

my $summary = "deduplication_summary.tsv";
open(OUT, ">$summary");
print OUT "Sample\\t";
my @headArr = ();
for my $mapOrder (@sortedOrderArray) {
    push (@headArr, @{$headerText{$mapOrder}});
}
my $headArrAll = join("\\t", @headArr);
print OUT "$headArrAll\\n";

foreach my $name (keys %tsv){
    my @rowArr = ();
    for my $mapOrder (@sortedOrderArray) {
        push (@rowArr, @{$tsv{$name}{$headerHash{$mapOrder}}});
    }
    my $rowArrAll = join("\\t", @rowArr);
    print OUT "$name\\t$rowArrAll\\n";
}
close(OUT);
'''
}

//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 240
    $CPU  = 1
    $MEMORY = 8
    $QUEUE = "short"
}
//* platform
//* autofill

process Sequential_Mapping_Module_Sequential_Mapping_Summary {

input:
 set val(name), file(bowfile) from g256_46_bowfiles10_g256_26
 val mate from g_347_mate11_g256_26
 val filtersList from g256_46_filter42_g256_26

output:
 file '*.tsv'  into g256_26_outputFileTSV00_g256_13
 val "sequential_mapping_sum"  into g256_26_name11_g256_13

errorStrategy 'retry'
maxRetries 2

shell:
'''
#!/usr/bin/env perl
open(my \$fh, '>', '!{name}.tsv');
print $fh "Sample\\tGroup\\tTotal Reads\\tReads After Sequential Mapping\\tUniquely Mapped\\tMultimapped\\tMapped\\n";
my @bowArray = split(' ', "!{bowfile}");
my $group= "\\t";
my @filterArray = (!{filtersList});
foreach my $bowitem(@bowArray) {
    # get mapping id
    my @bowAr = $bowitem.split("_");
    $bowCount = $bowAr[0] + -1;
    # if bowfiles ends with underscore (eg. bow_rRNA), parse rRNA as a group.
    my ($RDS_In, $RDS_After, $RDS_Uniq, $RDS_Multi, $ALGN_T, $a, $b, $aPer, $bPer)=(0, 0, 0, 0, 0, 0, 0, 0, 0);
    if ($bowitem =~ m/bow_([^\\.]+)$/){
        $group = "$1\\t";
        open(IN, $bowitem);
        my $i = 0;
        while(my $line=<IN>){
            chomp($line);
            $line=~s/^ +//;
            my @arr=split(/ /, $line);
            $RDS_In=$arr[0] if ($i=~/^1$/);
            # Reads After Filtering column depends on filtering type
            if ($i == 2){
                if ($filterArray[$bowCount] eq "Yes"){
                    $RDS_After=$arr[0];
                } else {
                    $RDS_After=$RDS_In;
                }
            }
            if ($i == 3){
                $a=$arr[0];
                $aPer=$arr[1];
                $aPer=~ s/([()])//g;
                $RDS_Uniq=$arr[0];
            }
            if ($i == 4){
                $b=$arr[0];
                $bPer=$arr[1];
                $bPer=~ s/([()])//g;
                $RDS_Multi=$arr[0];
            }
            $ALGN_T=($a+$b);
            $i++;
        }
        close(IN);
    } elsif ($bowitem =~ m/star_([^\\.]+)$/){
        $group = "$1\\t";
        open(IN2, $bowitem);
        my $multimapped;
		my $aligned;
		my $inputCount;
		chomp($inputCount = `cat $bowitem | grep 'Number of input reads' | awk '{sum+=\\$6} END {print sum}'`);
		chomp($uniqAligned = `cat $bowitem | grep 'Uniquely mapped reads number' | awk '{sum+=\\$6} END {print sum}'`);
		chomp($multimapped = `cat $bowitem | grep 'Number of reads mapped to multiple loci' | awk '{sum+=\\$9} END {print sum}'`);
		## Here we exclude "Number of reads mapped to too many loci" from multimapped reads since in bam file it called as unmapped.
		## Besides, these "too many loci" reads exported as unmapped reads from STAR.
		$RDS_In = int($inputCount);
		$RDS_Multi = int($multimapped);
        $RDS_Uniq = int($uniqAligned);
        $ALGN_T = $RDS_Uniq+$RDS_Multi;
		if ($filterArray[$bowCount] eq "Yes"){
            $RDS_After=$RDS_In-$ALGN_T;
        } else {
            $RDS_After=$RDS_In;
        }
    } elsif ($bowitem =~ m/bow1_([^\\.]+)$/){
        $group = "$1\\t";
        open(IN2, $bowitem);
        my $multimapped;
		my $aligned;
		my $inputCount;
		my $uniqAligned;
		chomp($inputCount = `cat $bowitem | grep '# reads processed:' | awk '{sum+=\\$4} END {print sum}'`);
		chomp($aligned = `cat $bowitem | grep '# reads with at least one reported alignment:' | awk '{sum+=\\$9} END {print sum}'`);
		chomp($uniqAligned = `cat $bowitem | grep '# unique mapped reads:' | awk '{sum+=\\$5} END {print sum}'`);
		## Here we exclude "Number of reads mapped to too many loci" from multimapped reads since in bam file it called as unmapped.
		## Besides, these "too many loci" reads exported as unmapped reads from STAR.
		$RDS_In = int($inputCount);
		$RDS_Multi = int($aligned) -int($uniqAligned);
		if ($RDS_Multi < 0 ){
		    $RDS_Multi = 0;
		}
        $RDS_Uniq = int($uniqAligned);
        $ALGN_T = int($aligned);
		if ($filterArray[$bowCount] eq "Yes"){
            $RDS_After=$RDS_In-$ALGN_T;
        } else {
            $RDS_After=$RDS_In;
        }
    }
    
    print $fh "!{name}\\t$group$RDS_In\\t$RDS_After\\t$RDS_Uniq\\t$RDS_Multi\\t$ALGN_T\\n";
}
close($fh);



'''

}


process Sequential_Mapping_Module_Merge_TSV_Files {

input:
 file tsv from g256_26_outputFileTSV00_g256_13.collect()
 val outputFileName from g256_26_name11_g256_13.collect()

output:
 file "${name}.tsv"  into g256_13_outputFileTSV00_g256_14

errorStrategy 'retry'
maxRetries 3

script:
name = outputFileName[0]
"""    
awk 'FNR==1 && NR!=1 {  getline; } 1 {print} ' *.tsv > ${name}.tsv
"""
}


process Sequential_Mapping_Module_Sequential_Mapping_Short_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /sequential_mapping_short_sum.tsv$/) "sequential_mapping_summary/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /sequential_mapping_detailed_sum.tsv$/) "sequential_mapping_summary/$filename"}
input:
 file mainSum from g256_13_outputFileTSV00_g256_14

output:
 file "sequential_mapping_short_sum.tsv"  into g256_14_outputFileTSV01_g_198
 file "sequential_mapping_detailed_sum.tsv"  into g256_14_outputFile11

errorStrategy 'retry'
maxRetries 2

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;

my @header;
my %all_rows;
my @seen_cols_short;
my @seen_cols_detailed;
my $ID_header;

chomp(my $contents = `ls *.tsv`);
my @files = split(/[\\n]+/, $contents);
foreach my $file (@files){
        open IN,"$file";
        my $line1 = <IN>;
        chomp($line1);
        ( $ID_header, my @h) = ( split("\\t", $line1) );
        my $totalHeader = $h[1];
        my $afterFilteringHeader = $h[2];
        my $uniqueHeader = $h[3];
        my $multiHeader = $h[4];
        my $mappedHeader = $h[5];
        push(@seen_cols_short, $totalHeader) unless grep{$_ eq $totalHeader} @seen_cols_short; #Total reads Header
        push(@seen_cols_detailed, $totalHeader) unless grep{$_ eq $totalHeader} @seen_cols_detailed; #Total reads Header

        my $n=0;
        while (my $line=<IN>) {
                
                chomp($line);
                my ( $ID, @fields ) = ( split("\\t", $line) ); 
                #SHORT
                push(@seen_cols_short, $fields[0]) unless grep{$_ eq $fields[0]} @seen_cols_short; #mapped element header
                $all_rows{$ID}{$fields[0]} = $fields[5];#Mapped Reads
                #Grep first line $fields[1] as total reads.
                if (!exists $all_rows{$ID}{$totalHeader}){    
                        $all_rows{$ID}{$totalHeader} = $fields[1];
                } 
                $all_rows{$ID}{$afterFilteringHeader} = $fields[2]; #only use last entry
                #DETAILED
                $uniqueHeadEach = "$fields[0] (${uniqueHeader})";
                $multiHeadEach = "$fields[0] (${multiHeader})";
                $mappedHeadEach = "$fields[0] (${mappedHeader})";
                push(@seen_cols_detailed, $mappedHeadEach) unless grep{$_ eq $mappedHeadEach} @seen_cols_detailed;
                push(@seen_cols_detailed, $uniqueHeadEach) unless grep{$_ eq $uniqueHeadEach} @seen_cols_detailed;
                push(@seen_cols_detailed, $multiHeadEach) unless grep{$_ eq $multiHeadEach} @seen_cols_detailed;
                $all_rows{$ID}{$mappedHeadEach} = $fields[5];
                $all_rows{$ID}{$uniqueHeadEach} = $fields[3];
                $all_rows{$ID}{$multiHeadEach} = $fields[4];
    }
    close IN;
    push(@seen_cols_short, $afterFilteringHeader) unless grep{$_ eq $afterFilteringHeader} @seen_cols_short; #After filtering Header
}


#print Dumper \\%all_rows;
#print Dumper \\%seen_cols_short;

printFiles("sequential_mapping_short_sum.tsv",@seen_cols_short,);
printFiles("sequential_mapping_detailed_sum.tsv",@seen_cols_detailed);


sub printFiles {
    my($summary, @cols_to_print) = @_;
    
    open OUT, ">$summary";
    print OUT join ("\\t", $ID_header,@cols_to_print),"\\n";
    foreach my $key ( keys %all_rows ) { 
        print OUT join ("\\t", $key, (map { $all_rows{$key}{$_} // '' } @cols_to_print)),"\\n";
        }
        close OUT;
}

'''


}

//* params.rsem_ref_using_star_index =  ""  //* @input
//* params.rsem_ref_using_bowtie2_index =  ""  //* @input
//* params.rsem_ref_using_bowtie_index =  ""  //* @input
//* @style @condition:{no_bam_output="false", output_genome_bam}, {no_bam_output="true"} @multicolumn:{no_bam_output, output_genome_bam}

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 4
    $MEMORY = 40
}
//* platform
if ($HOSTNAME == "hpc.umassmed.edu"){
    $CPU  = 4
    $MEMORY = 40
    $QUEUE = "long"
    $TIME = 1000
}
//* platform
//* autofill


process RSEM_module_RSEM {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /pipe.rsem.${name}$/) "rsem/$filename"}
input:
 val mate from g_347_mate10_g250_26
 set val(name), file(reads) from g256_46_reads01_g250_26
 file rsemIndex from g250_32_rsemIndex02_g250_26

output:
 file "pipe.rsem.${name}"  into g250_26_rsemOut00_g250_17, g250_26_rsemOut00_g250_21, g250_26_rsemOut01_g_177
 set val(name), file("pipe.rsem.*/*.genome.bam") optional true  into g250_26_bam_file11_g251_131, g250_26_bam_file10_g251_121, g250_26_bam_file10_g251_143

container "quay.io/viascientific/rsem:1.0"

when:
(params.run_RSEM && (params.run_RSEM == "yes")) || !params.run_RSEM

script:
RSEM_reference_type = params.RSEM_module_RSEM.RSEM_reference_type
RSEM_parameters = params.RSEM_module_RSEM.RSEM_parameters
no_bam_output = params.RSEM_module_RSEM.no_bam_output
output_genome_bam = params.RSEM_module_RSEM.output_genome_bam
sense_antisense = params.RSEM_module_RSEM.sense_antisense

nameAll = reads.toString()
nameArray = nameAll.split(' ')
def file2=""
if (nameAll.contains('.gz')) {
    file1 =  nameArray[0] - '.gz' 
    if (mate == "pair") {file2 =  nameArray[1] - '.gz'}
    runGzip = "ls *.gz | xargs -i echo gzip -df {} | sh"
} 

noBamText = ""
genome_BamText = ""

if (output_genome_bam.toString() != "false"){
	genome_BamText = "--output-genome-bam"
	// to prevent "two mates are aligned to two different transcripts" error, we use --star-output-genome-bam
	// so we need to sort bam file after rsem
	if (RSEM_reference_type == "star"){
		genome_BamText = "--star-output-genome-bam"
	}
}

if (no_bam_output.toString() != "false"){
    noBamText = "--no-bam-output"
    genome_BamText = ""
} 


refType = ""
finalReads = ""
if (RSEM_reference_type == "star"){
    refType = "--star --star-gzipped-read-file"
    runGzip = ''
    finalReads= "${reads}"
} else if (RSEM_reference_type == "bowtie2"){
    refType = "--bowtie2"
    finalReads= "${file1} ${file2}"
} else if (RSEM_reference_type == "bowtie"){
    refType = ""
    finalReads= "${file1} ${file2}"
}
"""
# basename=\$(basename ${rsemIndex}/*.ti | cut -d. -f1)
basename=\$(find -L ${rsemIndex} -type f -name "*.ti" -exec basename {} .ti \\; | sed 's/.ti\$//')
rsemRef=${rsemIndex}/\${basename}
$runGzip
mkdir -p pipe.rsem.${name}

if [ "${mate}" == "pair" ]; then
    echo "rsem-calculate-expression ${refType} -p ${task.cpus} ${RSEM_parameters} ${genome_BamText} ${noBamText} --paired-end ${finalReads} \${rsemRef} pipe.rsem.${name}/rsem.out.${name}"
    rsem-calculate-expression ${refType} -p ${task.cpus} ${RSEM_parameters} ${genome_BamText} ${noBamText} --paired-end ${finalReads}  \${rsemRef} pipe.rsem.${name}/rsem.out.${name}
	if [ "${sense_antisense}" == "Yes" ]; then
		 rsem-calculate-expression ${refType} -p ${task.cpus} ${RSEM_parameters} ${genome_BamText} ${noBamText} --forward-prob 1 --paired-end ${finalReads} \${rsemRef} pipe.rsem.${name}/rsem.out.forward.${name}
		 rsem-calculate-expression ${refType} -p ${task.cpus} ${RSEM_parameters} ${genome_BamText} ${noBamText} --forward-prob 0 --paired-end ${finalReads} \${rsemRef} pipe.rsem.${name}/rsem.out.reverse.${name}
	fi
else
    echo "rsem-calculate-expression ${refType} -p ${task.cpus} ${RSEM_parameters} ${genome_BamText} ${noBamText}  ${finalReads} \${rsemRef} pipe.rsem.${name}/rsem.out.${name}"
    rsem-calculate-expression ${refType} -p ${task.cpus} ${RSEM_parameters} ${genome_BamText} ${noBamText}  ${finalReads} \${rsemRef} pipe.rsem.${name}/rsem.out.${name}
	if [ "${sense_antisense}" == "Yes" ]; then
	    rsem-calculate-expression ${refType} -p ${task.cpus} ${RSEM_parameters} ${genome_BamText} ${noBamText} --forward-prob 1 ${finalReads} \${rsemRef} pipe.rsem.${name}/rsem.out.forward.${name}
    	rsem-calculate-expression ${refType} -p ${task.cpus} ${RSEM_parameters} ${genome_BamText} ${noBamText} --forward-prob 0 ${finalReads} \${rsemRef} pipe.rsem.${name}/rsem.out.reverse.${name}
	fi
fi
## --star-output-genome-bam flag creates unsorted STAR genome bam
if [ -e "pipe.rsem.${name}/rsem.out.${name}.STAR.genome.bam" ] ; then
	mv pipe.rsem.${name}/rsem.out.${name}.STAR.genome.bam pipe.rsem.${name}/rsem.out.${name}.genome.bam
fi
if [ -e "pipe.rsem.${name}/rsem.out.${name}.genome.bam" ] ; then
    mv pipe.rsem.${name}/rsem.out.${name}.genome.bam pipe.rsem.${name}/rsem.out.${name}.genome.unsorted.bam
    samtools sort -o pipe.rsem.${name}/rsem.out.${name}.genome.bam pipe.rsem.${name}/rsem.out.${name}.genome.unsorted.bam
    rm pipe.rsem.${name}/rsem.out.${name}.genome.unsorted.bam
fi


"""

}

//* autofill
//* platform
//* platform
//* autofill

process RSEM_module_RSEM_Count {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "rsem_summary/$filename"}
input:
 file rsemOut from g250_26_rsemOut00_g250_21.collect()

output:
 file "*.tsv"  into g250_21_outputFile00_g292_25, g250_21_outputFile00_g292_24

shell:
if (params.gtf){
	gtf=params.gtf
}
if (params.senseantisense){
	senseantisense = params.senseantisense
}
'''
#!/usr/bin/env perl

my %tf = (
        expected_count => 4,
        tpm => 5,
        fpkm => 6,
    );

my $indir = $ENV{'PWD'};
$outdir = $ENV{'PWD'};

my @sense_antisense = ("", ".forward", ".reverse");
my @gene_iso_ar = ("genes", "isoforms");
my @tpm_fpkm_expectedCount_ar = ("expected_count", "tpm");
for($m = 0; $m <= $#sense_antisense; $m++) {
	my $sa = $sense_antisense[$m];
	for($l = 0; $l <= $#gene_iso_ar; $l++) {
	    my $gene_iso = $gene_iso_ar[$l];
	    for($ll = 0; $ll <= $#tpm_fpkm_expectedCount_ar; $ll++) {
	        my $tpm_fpkm_expectedCount = $tpm_fpkm_expectedCount_ar[$ll];
	
	        opendir D, $indir or die "Could not open $indir\n";
	        my @alndirs = sort { $a cmp $b } grep /^pipe/, readdir(D);
	        closedir D;
	    
	        my @a=();
	        my %b=();
	        my %c=();
	        my $i=0;
	        my $saexist=0;
	        foreach my $d (@alndirs){ 
	            my $dir = "${indir}/$d";
	            print $d."\n";
	            my $libname=$d;
	            $libname=~s/pipe\\.rsem\\.//;
	    
	            $i++;
	            $a[$i]=$libname;
	            if (-e "${dir}/rsem.out$sa.$libname.$gene_iso.results"){
	            	$saexist=1;
	         
		            open IN,"${dir}/rsem.out$sa.$libname.$gene_iso.results";
		            $_=<IN>;
		            while(<IN>)
		            {
		                my @v=split; 
		                $b{$v[0]}{$i}=$v[$tf{$tpm_fpkm_expectedCount}];
		                $c{$v[0]}=$v[1];
		            }
		            close IN;
	            }
	        }
	        if ($saexist==1){
		        my $outfile="${indir}/${gene_iso}${sa}_expression_"."$tpm_fpkm_expectedCount".".tsv";
		        open OUT, ">$outfile";
		        if ($gene_iso ne "isoforms") {
		            print OUT "gene\ttranscript";
		        } else {
		            print OUT "transcript\tgene";
		        }
		    
		        for(my $j=1;$j<=$i;$j++) {
		            print OUT "\t$a[$j]";
		        }
		        print OUT "\n";
		    
		        foreach my $key (keys %b) {
		            print OUT "$key\t$c{$key}";
		            for(my $j=1;$j<=$i;$j++){
		                print OUT "\t$b{$key}{$j}";
		            }
		            print OUT "\n";
		        }
		        close OUT;
		        if ($sa eq ".reverse"){
		        	my $com = "perl !{senseantisense} !{gtf} ${indir}/${gene_iso}.forward_expression_${tpm_fpkm_expectedCount}.tsv ${indir}/${gene_iso}${sa}_expression_${tpm_fpkm_expectedCount}.tsv";
		        	`$com`;
		        	$com = "rm -rf ${indir}/${gene_iso}.forward_expression_${tpm_fpkm_expectedCount}.tsv ${indir}/${gene_iso}${sa}_expression_${tpm_fpkm_expectedCount}.tsv";
		        	`$com`;
		        }
	        }
	    }
	}
}
'''
}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_RSEM_Prepare_DESeq2 {

input:
 file counts from g250_21_outputFile00_g292_24
 file groups_file from g_295_1_g292_24
 file compare_file from g_294_2_g292_24
 val run_DESeq2 from g_293_3_g292_24

output:
 file "DE_reports"  into g292_24_outputFile00_g292_37
 val "_des"  into g292_24_postfix10_g292_33
 file "DE_reports/outputs/*_all_deseq2_results.tsv"  into g292_24_outputFile21_g292_33

container 'quay.io/viascientific/de_module:4.0'

when:
run_DESeq2 == 'yes'

script:

feature_type = params.DE_module_RSEM_Prepare_DESeq2.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_RSEM_Prepare_DESeq2.include_distribution
include_all2all = params.DE_module_RSEM_Prepare_DESeq2.include_all2all
include_pca = params.DE_module_RSEM_Prepare_DESeq2.include_pca

filter_type = params.DE_module_RSEM_Prepare_DESeq2.filter_type
min_count = params.DE_module_RSEM_Prepare_DESeq2.min_count
min_samples = params.DE_module_RSEM_Prepare_DESeq2.min_samples
min_counts_per_sample = params.DE_module_RSEM_Prepare_DESeq2.min_counts_per_sample
excluded_events = params.DE_module_RSEM_Prepare_DESeq2.excluded_events

include_batch_correction = params.DE_module_RSEM_Prepare_DESeq2.include_batch_correction
batch_correction_column = params.DE_module_RSEM_Prepare_DESeq2.batch_correction_column
batch_correction_group_column = params.DE_module_RSEM_Prepare_DESeq2.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_RSEM_Prepare_DESeq2.batch_normalization_algorithm

transformation = params.DE_module_RSEM_Prepare_DESeq2.transformation
pca_color = params.DE_module_RSEM_Prepare_DESeq2.pca_color
pca_shape = params.DE_module_RSEM_Prepare_DESeq2.pca_shape
pca_fill = params.DE_module_RSEM_Prepare_DESeq2.pca_fill
pca_transparency = params.DE_module_RSEM_Prepare_DESeq2.pca_transparency
pca_label = params.DE_module_RSEM_Prepare_DESeq2.pca_label

include_deseq2 = params.DE_module_RSEM_Prepare_DESeq2.include_deseq2
input_mode = params.DE_module_RSEM_Prepare_DESeq2.input_mode
design = params.DE_module_RSEM_Prepare_DESeq2.design
fitType = params.DE_module_RSEM_Prepare_DESeq2.fitType
use_batch_corrected_in_DE = params.DE_module_RSEM_Prepare_DESeq2.use_batch_corrected_in_DE
apply_shrinkage = params.DE_module_RSEM_Prepare_DESeq2.apply_shrinkage
shrinkage_type = params.DE_module_RSEM_Prepare_DESeq2.shrinkage_type
include_volcano = params.DE_module_RSEM_Prepare_DESeq2.include_volcano
include_ma = params.DE_module_RSEM_Prepare_DESeq2.include_ma
include_heatmap = params.DE_module_RSEM_Prepare_DESeq2.include_heatmap

padj_significance_cutoff = params.DE_module_RSEM_Prepare_DESeq2.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_RSEM_Prepare_DESeq2.fc_significance_cutoff
padj_floor = params.DE_module_RSEM_Prepare_DESeq2.padj_floor
fc_ceiling = params.DE_module_RSEM_Prepare_DESeq2.fc_ceiling

convert_names = params.DE_module_RSEM_Prepare_DESeq2.convert_names
count_file_names = params.DE_module_RSEM_Prepare_DESeq2.count_file_names
converted_name = params.DE_module_RSEM_Prepare_DESeq2.converted_name
org_db = params.DE_module_RSEM_Prepare_DESeq2.org_db
num_labeled = params.DE_module_RSEM_Prepare_DESeq2.num_labeled
highlighted_genes = params.DE_module_RSEM_Prepare_DESeq2.highlighted_genes
include_volcano_highlighted = params.DE_module_RSEM_Prepare_DESeq2.include_volcano_highlighted
include_ma_highlighted = params.DE_module_RSEM_Prepare_DESeq2.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name, org_db},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm, use_batch_corrected_in_DE},{include_batch_correction="false"},{include_deseq2="true", design, fitType, apply_shrinkage, shrinkage_type, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, org_db, num_labeled, highlighted_genes},{include_deseq2="false"},{apply_shrinkage="true", shrinkage_type},{apply_shrinkage="false"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples, min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{design, fitType, use_batch_corrected_in_DE, apply_shrinkage, shrinkage_type},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name, org_db},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_deseq2 = include_deseq2 == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
apply_shrinkage = apply_shrinkage == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir reports
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_DESeq2.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-DESeq2 ${include_deseq2} --input-mode ${input_mode} --design '${design}' --fitType ${fitType} --use-batch-correction-in-DE ${use_batch_corrected_in_DE} --apply-shrinkage ${apply_shrinkage} --shrinkage-type ${shrinkage_type} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --org-db ${org_db} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_RSEM_Prepare_LimmaVoom {

input:
 file counts from g250_21_outputFile00_g292_25
 file groups_file from g_295_1_g292_25
 file compare_file from g_294_2_g292_25
 val run_limmaVoom from g_356_3_g292_25

output:
 file "DE_reports"  into g292_25_outputFile00_g292_39
 val "_lv"  into g292_25_postfix10_g292_41
 file "DE_reports/outputs/*_all_limmaVoom_results.tsv"  into g292_25_outputFile21_g292_41

container 'quay.io/viascientific/de_module:4.0'

when:
run_limmaVoom == 'yes'

script:

feature_type = params.DE_module_RSEM_Prepare_LimmaVoom.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_RSEM_Prepare_LimmaVoom.include_distribution
include_all2all = params.DE_module_RSEM_Prepare_LimmaVoom.include_all2all
include_pca = params.DE_module_RSEM_Prepare_LimmaVoom.include_pca

filter_type = params.DE_module_RSEM_Prepare_LimmaVoom.filter_type
min_count = params.DE_module_RSEM_Prepare_LimmaVoom.min_count
min_samples = params.DE_module_RSEM_Prepare_LimmaVoom.min_samples
min_counts_per_sample = params.DE_module_RSEM_Prepare_LimmaVoom.min_counts_per_sample
excluded_events = params.DE_module_RSEM_Prepare_LimmaVoom.excluded_events

include_batch_correction = params.DE_module_RSEM_Prepare_LimmaVoom.include_batch_correction
batch_correction_column = params.DE_module_RSEM_Prepare_LimmaVoom.batch_correction_column
batch_correction_group_column = params.DE_module_RSEM_Prepare_LimmaVoom.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_RSEM_Prepare_LimmaVoom.batch_normalization_algorithm

transformation = params.DE_module_RSEM_Prepare_LimmaVoom.transformation
pca_color = params.DE_module_RSEM_Prepare_LimmaVoom.pca_color
pca_shape = params.DE_module_RSEM_Prepare_LimmaVoom.pca_shape
pca_fill = params.DE_module_RSEM_Prepare_LimmaVoom.pca_fill
pca_transparency = params.DE_module_RSEM_Prepare_LimmaVoom.pca_transparency
pca_label = params.DE_module_RSEM_Prepare_LimmaVoom.pca_label

include_limma = params.DE_module_RSEM_Prepare_LimmaVoom.include_limma
use_batch_corrected_in_DE = params.DE_module_RSEM_Prepare_LimmaVoom.use_batch_corrected_in_DE
normalization_method = params.DE_module_RSEM_Prepare_LimmaVoom.normalization_method
logratioTrim = params.DE_module_RSEM_Prepare_LimmaVoom.logratioTrim
sumTrim = params.DE_module_RSEM_Prepare_LimmaVoom.sumTrim
Acutoff = params.DE_module_RSEM_Prepare_LimmaVoom.Acutoff
doWeighting = params.DE_module_RSEM_Prepare_LimmaVoom.doWeighting
p = params.DE_module_RSEM_Prepare_LimmaVoom.p
include_volcano = params.DE_module_RSEM_Prepare_LimmaVoom.include_volcano
include_ma = params.DE_module_RSEM_Prepare_LimmaVoom.include_ma
include_heatmap = params.DE_module_RSEM_Prepare_LimmaVoom.include_heatmap

padj_significance_cutoff = params.DE_module_RSEM_Prepare_LimmaVoom.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_RSEM_Prepare_LimmaVoom.fc_significance_cutoff
padj_floor = params.DE_module_RSEM_Prepare_LimmaVoom.padj_floor
fc_ceiling = params.DE_module_RSEM_Prepare_LimmaVoom.fc_ceiling

convert_names = params.DE_module_RSEM_Prepare_LimmaVoom.convert_names
count_file_names = params.DE_module_RSEM_Prepare_LimmaVoom.count_file_names
converted_name = params.DE_module_RSEM_Prepare_LimmaVoom.converted_name
num_labeled = params.DE_module_RSEM_Prepare_LimmaVoom.num_labeled
highlighted_genes = params.DE_module_RSEM_Prepare_LimmaVoom.highlighted_genes
include_volcano_highlighted = params.DE_module_RSEM_Prepare_LimmaVoom.include_volcano_highlighted
include_ma_highlighted = params.DE_module_RSEM_Prepare_LimmaVoom.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm,use_batch_corrected_in_DE},{include_batch_correction="false"},{include_limma="true", normalization_method, logratioTrim, sumTrim, doWeighting, Acutoff, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, num_labeled, highlighted_genes},{include_limma="false"},{normalization_method="TMM", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="TMMwsp", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="RLE"},{normalization_method="upperquartile", p},{normalization_method="none"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples,min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_limma, use_batch_corrected_in_DE},{normalization_method,logratioTrim,sumTrim,doWeighting,Acutoff,p},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_limma = include_limma == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

doWeighting = doWeighting == 'true' ? 'TRUE' : 'FALSE'
TMM_args = normalization_method.equals('TMM') || normalization_method.equals('TMMwsp') ? '--logratio-trim ' + logratioTrim + ' --sum-trim ' + sumTrim + ' --do-weighting ' + doWeighting + ' --A-cutoff="' + Acutoff + '"' : ''
upperquartile_args = normalization_method.equals('upperquartile') ? '--p ' + p : ''

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_limmaVoom.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-limma ${include_limma} \
--use-batch-correction-in-DE ${use_batch_corrected_in_DE} --normalization-method ${normalization_method} ${TMM_args} ${upperquartile_args} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}

//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 30
    $CPU  = 1
    $MEMORY = 10
    $QUEUE = "short"
}
//* platform
//* autofill

process RSEM_module_RSEM_Alignment_Summary {

input:
 file rsemDir from g250_26_rsemOut00_g250_17.collect()

output:
 file "rsem_alignment_sum.tsv"  into g250_17_outputFileTSV03_g_198

shell:
'''

#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;
my $indir = $ENV{'PWD'};

opendir D, $indir or die "Could not open $indir";
my @alndirs = sort { $a cmp $b } grep /^pipe/, readdir(D);
closedir D;

my @a=();
my %b=();
my %c=();
my $i=0;
my @headers = ();
my %tsv;
foreach my $d (@alndirs){
    my $dir = "${indir}/$d";
    my $libname=$d;
    $libname=~s/pipe\\.rsem\\.//;
    my $multimapped;
    my $aligned;
    my $total;
    
    chomp($total = `awk 'NR == 1 {print \\$4}' ${dir}/rsem.out.$libname.stat/rsem.out.$libname.cnt`);
    chomp($aligned = `awk 'NR == 1 {print \\$2}' ${dir}/rsem.out.$libname.stat/rsem.out.$libname.cnt`);
    chomp($multimapped = `awk 'NR == 2 {print \\$2}' ${dir}/rsem.out.$libname.stat/rsem.out.$libname.cnt`);
    $tsv{$libname}=[$libname, $total];
    push(@{$tsv{$libname}}, $multimapped);
    push(@{$tsv{$libname}}, (int($aligned) - int($multimapped))."");
}


push(@headers, "Sample");
push(@headers, "Total Reads");
push(@headers, "Multimapped Reads Aligned (RSEM)");
push(@headers, "Unique Aligned Reads (RSEM)");


my @keys = keys %tsv;
my $summary = "rsem_alignment_sum.tsv";
my $header_string = join("\\t", @headers);
`echo "$header_string" > $summary`;
foreach my $key (@keys){
    my $values = join("\\t", @{ $tsv{$key} });
        `echo "$values" >> $summary`;
}
'''
}


process BAM_Analysis_Module_RSEM_bam_sort_index {

input:
 set val(name), file(bam) from g250_26_bam_file10_g251_143

output:
 set val(name), file("bam/*.bam"), file("bam/*.bam.bai")  into g251_143_bam_bai00_g251_134, g251_143_bam_bai00_g251_142

when:
params.run_BigWig_Conversion == "yes" || params.run_RSeQC == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}

"""
$runSamtools
mkdir -p bam
mv ${name}_sorted.bam ${name}_sorted.bam.bai bam/.
"""
}



//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 30
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_RSEM_UCSC_BAM2BigWig_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.bw$/) "bigwig_rsem/$filename"}
input:
 set val(name), file(bam), file(bai) from g251_143_bam_bai00_g251_142
 file genomeSizes from g245_54_genomeSizes21_g251_142

output:
 file "*.bw" optional true  into g251_142_outputFileBw00
 file "publish/*.bw" optional true  into g251_142_publishBw10_g251_145

container 'quay.io/biocontainers/deeptools:3.5.4--pyhdfd78af_1'

when:
params.run_BigWig_Conversion == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = ""
    nameFinal = nameAll
} else {
    runSamtools = "mv $bam ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
deeptools_parameters = params.BAM_Analysis_Module_RSEM_UCSC_BAM2BigWig_converter.deeptools_parameters
visualize_bigwig_in_reports = params.BAM_Analysis_Module_RSEM_UCSC_BAM2BigWig_converter.visualize_bigwig_in_reports

"""
$runSamtools
bamCoverage ${deeptools_parameters}  -b ${nameFinal} -o ${name}.bw 

if [ "${visualize_bigwig_in_reports}" == "yes" ]; then
	mkdir -p publish
	mv ${name}.bw publish/.
fi
"""

}



process BAM_Analysis_Module_RSEM_Genome_Browser {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "genome_browser_rsem/$filename"}
input:
 file bigwigs from g251_142_publishBw10_g251_145.collect()
 file group_file from g_295_1_g251_145

output:
 file "*"  into g251_145_bigWig_file00

container 'quay.io/viascientific/python-basics:2.0'


script:
try {
    myVariable = group_file
} catch (MissingPropertyException e) {
    group_file = ""
}

genome_build_short= ""
if (params.genome_build == "mousetest_mm10"){
    genome_build_short= "mm10"
} else if (params.genome_build == "human_hg19_refseq"){
    genome_build_short = "hg19"
} else if (params.genome_build == "human_hg38_gencode_v28"){
    genome_build_short = "hg38"
} else if (params.genome_build == "human_hg38_gencode_v34"){
    genome_build_short = "hg38"
} else if (params.genome_build == "mouse_mm10_refseq"){
    genome_build_short = "mm10"
} else if (params.genome_build == "mouse_mm10_gencode_m25"){
    genome_build_short = "mm10"
} else if (params.genome_build == "rat_rn6_refseq"){
    genome_build_short = "rn6"
} else if (params.genome_build == "rat_rn6_ensembl_v86"){
    genome_build_short = "rn6"
} else if (params.genome_build == "zebrafish_GRCz11_ensembl_v95"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_refseq"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_v4.3.2"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "c_elegans_ce11_ensembl_ws245"){
    genome_build_short = "ce11"
} else if (params.genome_build == "d_melanogaster_dm6_refseq"){
    genome_build_short = "dm6"
} else if (params.genome_build == "s_cerevisiae_sacCer3_refseq"){
    genome_build_short = "sacCer3"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v31"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v51"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "e_coli_ASM584v2_refseq"){
    genome_build_short = "ASM584v2"
} else if (params.genome_build == "dog_canFam3_refseq"){
    genome_build_short = "canFam3"
} 

"""

#!/usr/bin/env python

import glob,json,os,sys,csv,random,shutil,requests,subprocess

def check_url_existence(url):
    try:
        response = requests.head(url, allow_redirects=True)
        # Check if the status code is in the success range (200-399)
        return 200 <= response.status_code < 400
    except requests.ConnectionError:
        return False

def get_lib_val(str, lib):
    for key, value in lib.items():
        if key.lower() in str.lower():
            return value
    return None

def find_and_move_folders_with_bw_files(start_dir):
    bigwig_dir = "bigwigs"
    if os.path.exists(bigwig_dir)!=True:
        os.makedirs(bigwig_dir)
    subprocess.getoutput("cp ${bigwigs} bigwigs/. ")

def Generate_HubFile(groupfile):
    # hub.txt
    Hub = open("hub.txt", "w")
    Hub.write("hub UCSCHub \\n")
    Hub.write("shortLabel UCSCHub \\n")
    Hub.write("longLabel UCSCHub \\n")
    Hub.write("genomesFile genomes.txt \\n")
    Hub.write("email support@viascientific.com \\n")
    Hub.write("\\n")
    Hub.close()

    # genomes.txt
    genomes = open("genomes.txt", "w")
    genomes.write("genome ${genome_build_short} \\n")
    genomes.write("trackDb bigwigs/trackDb.txt \\n")
    genomes.write("\\n")
    genomes.close()
    #trackDb = open("bigwigs/trackDb.txt", "w")
    path = r'bigwigs/*.bw'
    files = glob.glob(path)
    files.sort()
    sample={}
    for i in files:
        temp=i.split('.')[0]
        temp=temp.replace('bigwigs/','')
        if temp in sample.keys():
            sample[temp].append(i.replace('bigwigs/',''))
        else:
            sample[temp]=[]
            sample[temp].append(i.replace('bigwigs/',''))
    second_layer_indent=" "*2
    third_layer_indent = " " * 14
    if groupfile == "" or os.stat(groupfile).st_size == 0:
        #No GroupFile
        trackDb = open("bigwigs/trackDb.txt", "w")
        for i in sample.keys():
            trackDb.write('track %s\\n' %i)
            trackDb.write('shortLabel %s\\n' %i)
            trackDb.write('longLabel %s\\n' %i)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for j in sample[i]:
                trackDb.write(second_layer_indent+'track %s\\n' % j)
                trackDb.write(second_layer_indent+'bigDataUrl %s\\n' % j)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %i)
                trackDb.write('\\n')
        trackDb.close()
    else:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_to_sample={}
        for cond in condition:
            if cond not in condition_to_sample.keys():
                condition_to_sample[cond]=[]
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
            else:
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
        trackDb = open("bigwigs/trackDb.txt", "w")
        for cond in condition:
            trackDb.write('track %s\\n' %cond)
            trackDb.write('shortLabel %s\\n' %cond)
            trackDb.write('longLabel %s\\n' %cond)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for samp in condition_to_sample[cond]:
                trackDb.write(second_layer_indent+'track %s\\n' % samp)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'view Signal \\n')
                trackDb.write(second_layer_indent+'visibility full \\n')
                trackDb.write(second_layer_indent+'viewLimits 0:20 \\n')
                trackDb.write(second_layer_indent+'autoScale on \\n')
                trackDb.write(second_layer_indent+'maxHeightPixels 128:20:8 \\n')
                trackDb.write(second_layer_indent+'configurable on \\n')
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %cond)
                trackDb.write('\\n')
                for file in sample[samp]:
                    trackDb.write(third_layer_indent + 'track %s\\n' % file)
                    trackDb.write(third_layer_indent + 'bigDataUrl %s\\n' % file)
                    trackDb.write(third_layer_indent + 'shortLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'longLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'type bigWig\\n')
                    trackDb.write(third_layer_indent+'parent %s\\n' %samp)

                    trackDb.write('\\n')

        trackDb.close()


def Generating_Json_files(groupfile):
    publishWebDir = '{{DNEXT_WEB_REPORT_DIR}}/' + 'genome_browser_rsem' + "/bigwigs"
    locusLib = {}
    # MYC locations
    locusLib["hg19"] = "chr8:128,746,315-128,755,680"
    locusLib["hg38"] = "chr8:127,733,434-127,744,951"
    locusLib["mm10"] = "chr15:61,983,341-61,992,361"
    locusLib["rn6"] = "chr7:102,584,313-102,593,240"
    locusLib["dm6"] = "chrX:3,371,159-3,393,697"
    locusLib["canFam3"] = "chr13:25,198,772-25,207,309"

    cytobandLib = {}
    cytobandLib["hg19"] = "https://igv-genepattern-org.s3.amazonaws.com/genomes/seq/hg19/cytoBand.txt"
    cytobandLib["hg38"] = "https://s3.amazonaws.com/igv.org.genomes/hg38/annotations/cytoBandIdeo.txt.gz"
    cytobandLib["mm10"] = "https://s3.amazonaws.com/igv.broadinstitute.org/annotations/mm10/cytoBandIdeo.txt.gz"
    cytobandLib["rn6"] = "https://s3.amazonaws.com/igv.org.genomes/rn6/cytoBand.txt.gz"
    cytobandLib["dm6"] = "https://s3.amazonaws.com/igv.org.genomes/dm6/cytoBandIdeo.txt.gz"
    cytobandLib["ce11"] = "https://s3.amazonaws.com/igv.org.genomes/ce11/cytoBandIdeo.txt.gz"
    cytobandLib["canFam3"] = "https://s3.amazonaws.com/igv.org.genomes/canFam3/cytoBandIdeo.txt.gz"

    # Get the basename of the original path
    gtf_source_base_name = os.path.basename("${params.gtf_source}")
    gtf_source_sorted = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz")
    print(gtf_source_sorted)
    gtf_source_sorted_index = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz.tbi")
    print(gtf_source_sorted_index)


    data = {}
    data["reference"] = {}
    data["reference"]["id"] = "${params.genome_build}"
    data["reference"]["name"] = "${params.genome_build}"
    data["reference"]["fastaURL"] = "${params.genome_source}"
    data["reference"]["indexURL"] = "${params.genome_source}.fai"
    cytobandurl = get_lib_val("${params.genome_build}", cytobandLib)
    locusStr = get_lib_val("${params.genome_build}", locusLib)
    if cytobandurl is not None:
        data["reference"]["cytobandURL"] = cytobandurl
    if locusStr is not None:
        data["locus"] = []
        data["locus"].append(locusStr)
    data["tracks"] = []
    # prepare gtf Track
    gtfTrack = {}
    gtfTrack["name"] = "${params.genome_build}"
    gtfTrack["gtf"] = "gtf"
    if check_url_existence(gtf_source_sorted):
        gtfTrack["url"] = gtf_source_sorted
    else:
        gtfTrack["url"] = "${params.gtf_source}"
    if check_url_existence(gtf_source_sorted_index):
        gtfTrack["indexURL"] = gtf_source_sorted_index

    # prepare cytobands Track
    if groupfile and os.path.isfile(groupfile) and os.stat(groupfile).st_size != 0:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_color_dict = dict()
        for cond in condition:
            r = lambda: random.randint(0, 255)
            condition_color_dict[cond] = '#%02X%02X%02X' % (r(), r(), r())

    # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                if basename in data_parsed:
                    newdata[file] = data_parsed[basename]
                    newdata[file]["fullname"] = file

        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": condition_color_dict[newdata[j]['group']],
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()
    else:

        # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                newdata[file] = basename
                newdata[file] = file
        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            r = lambda: random.randint(0, 255)
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": '#%02X%02X%02X' % (r(), r(), r()),
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()

if __name__ == "__main__":
    find_and_move_folders_with_bw_files(".")
    Generate_HubFile(groupfile="${group_file}")
    Generating_Json_files(groupfile="${group_file}")

"""
}

//* params.bed =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 10
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_RSEM_RSeQC {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "rseqc_rsem/$filename"}
input:
 set val(name), file(bam), file(bai) from g251_143_bam_bai00_g251_134
 file bed from g245_54_bed31_g251_134
 val mate from g_347_mate12_g251_134

output:
 file "*"  into g251_134_outputFileOut07_g_177

container 'quay.io/viascientific/rseqc:1.0'

when:
(params.run_RSeQC && (params.run_RSeQC == "yes")) || !params.run_RSeQC

script:
run_bam_stat = params.BAM_Analysis_Module_RSEM_RSeQC.run_bam_stat
run_read_distribution = params.BAM_Analysis_Module_RSEM_RSeQC.run_read_distribution
run_inner_distance = params.BAM_Analysis_Module_RSEM_RSeQC.run_inner_distance
run_junction_annotation = params.BAM_Analysis_Module_RSEM_RSeQC.run_junction_annotation
run_junction_saturation = params.BAM_Analysis_Module_RSEM_RSeQC.run_junction_saturation
//run_geneBody_coverage and run_infer_experiment needs subsampling
run_geneBody_coverage = params.BAM_Analysis_Module_RSEM_RSeQC.run_geneBody_coverage
run_infer_experiment = params.BAM_Analysis_Module_RSEM_RSeQC.run_infer_experiment
"""
if [ "$run_bam_stat" == "true" ]; then bam_stat.py  -i ${bam} > ${name}.bam_stat.txt; fi
if [ "$run_read_distribution" == "true" ]; then read_distribution.py  -i ${bam} -r ${bed}> ${name}.read_distribution.out; fi


if [ "$run_infer_experiment" == "true" -o "$run_geneBody_coverage" == "true" ]; then
	numAlignedReads=\$(samtools view -c -F 4 $bam)

	if [ "\$numAlignedReads" -gt 1000000 ]; then
    	echo "Read number is greater than 1000000. Subsampling..."
    	finalRead=1000000
    	fraction=\$(samtools idxstats  $bam | cut -f3 | awk -v ct=\$finalRead 'BEGIN {total=0} {total += \$1} END {print ct/total}')
    	samtools view -b -s \${fraction} $bam > ${name}_sampled.bam
    	samtools index ${name}_sampled.bam
    	if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i ${name}_sampled.bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i ${name}_sampled.bam  -r $bed -o ${name}; fi
	else
		if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i $bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i $bam  -r $bed -o ${name}; fi
	fi

fi


if [ "${mate}" == "pair" ]; then
	if [ "$run_inner_distance" == "true" ]; then inner_distance.py -i $bam  -r $bed -o ${name}.inner_distance > stdout.txt; fi
	if [ "$run_inner_distance" == "true" ]; then head -n 2 stdout.txt > ${name}.inner_distance_mean.txt; fi
fi
if [ "$run_junction_annotation" == "true" ]; then junction_annotation.py -i $bam  -r $bed -o ${name}.junction_annotation 2> ${name}.junction_annotation.log; fi
if [ "$run_junction_saturation" == "true" ]; then junction_saturation.py -i $bam  -r $bed -o ${name}; fi
if [ -e class.log ] ; then mv class.log ${name}_class.log; fi
if [ -e log.txt ] ; then mv log.txt ${name}_log.txt; fi
if [ -e stdout.txt ] ; then mv stdout.txt ${name}_stdout.txt; fi


"""

}

//* params.pdfbox_path =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 32
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 240
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "short"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 500
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
}
//* platform
//* autofill

process BAM_Analysis_Module_RSEM_Picard {

input:
 set val(name), file(bam) from g250_26_bam_file10_g251_121

output:
 file "*_metrics"  into g251_121_outputFileOut00_g251_82
 file "results/*.pdf"  into g251_121_outputFilePdf12_g251_82

container 'quay.io/viascientific/picard:1.0'

when:
(params.run_Picard_CollectMultipleMetrics && (params.run_Picard_CollectMultipleMetrics == "yes")) || !params.run_Picard_CollectMultipleMetrics

script:
"""
picard CollectMultipleMetrics OUTPUT=${name}_multiple.out VALIDATION_STRINGENCY=LENIENT INPUT=${bam}
mkdir results && java -jar ${params.pdfbox_path} PDFMerger *.pdf results/${name}_multi_metrics.pdf
"""
}


process BAM_Analysis_Module_RSEM_Picard_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "picard_summary_rsem/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "rseqc_summary_rsem/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /results\/.*.pdf$/) "picard_summary_pdf_rsem/$filename"}
input:
 file picardOut from g251_121_outputFileOut00_g251_82.collect()
 val mate from g_347_mate11_g251_82
 file picardPdf from g251_121_outputFilePdf12_g251_82.collect()

output:
 file "*.tsv"  into g251_82_outputFileTSV00
 file "results/*.pdf"  into g251_82_outputFilePdf11

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage; 
use Data::Dumper;

runCommand("mkdir results && mv *.pdf results/. ");

my $indir = $ENV{'PWD'};
my $outd = $ENV{'PWD'};
my @files = ();
my @outtypes = ("CollectRnaSeqMetrics", "alignment_summary_metrics", "base_distribution_by_cycle_metrics", "insert_size_metrics", "quality_by_cycle_metrics", "quality_distribution_metrics" );

foreach my $outtype (@outtypes)
{
my $ext="_multiple.out";
$ext.=".$outtype" if ($outtype ne "CollectRnaSeqMetrics");
@files = <$indir/*$ext>;

my @rowheaders=();
my @libs=();
my %metricvals=();
my %histvals=();

my $pdffile="";
my $libname="";
foreach my $d (@files){
  my $libname=basename($d, $ext);
  print $libname."\\n";
  push(@libs, $libname); 
  getMetricVals($d, $libname, \\%metricvals, \\%histvals, \\@rowheaders);
}

my $sizemetrics = keys %metricvals;
write_results("$outd/$outtype.stats.tsv", \\@libs,\\%metricvals, \\@rowheaders, "metric") if ($sizemetrics>0);
my $sizehist = keys %histvals;
write_results("$outd/$outtype.hist.tsv", \\@libs,\\%histvals, "none", "nt") if ($sizehist>0);

}

sub write_results
{
  my ($outfile, $libs, $vals, $rowheaders, $name )=@_;
  open(OUT, ">$outfile");
  print OUT "$name\\t".join("\\t", @{$libs})."\\n";
  my $size=0;
  $size=scalar(@{${$vals}{${$libs}[0]}}) if(exists ${$libs}[0] and exists ${$vals}{${$libs}[0]} );
  
  for (my $i=0; $i<$size;$i++)
  { 
    my $rowname=$i;
    $rowname = ${$rowheaders}[$i] if ($name=~/metric/);
    print OUT $rowname;
    foreach my $lib (@{$libs})
    {
      print OUT "\\t".${${$vals}{$lib}}[$i];
    } 
    print OUT "\\n";
  }
  close(OUT);
}

sub getMetricVals{
  my ($filename, $libname, $metricvals, $histvals,$rowheaders)=@_;
  if (-e $filename){
     my $nextisheader=0;
     my $nextisvals=0;
     my $nexthist=0;
     open(IN, $filename);
     while(my $line=<IN>)
     {
       chomp($line);
       @{$rowheaders}=split(/\\t/, $line) if ($nextisheader && !scalar(@{$rowheaders})); 
       if ($nextisvals) {
         @{${$metricvals}{$libname}}=split(/\\t/, $line);
         $nextisvals=0;
       }
       if($nexthist){
          my @vals=split(/[\\s\\t]+/,$line); 
          push(@{${$histvals}{$libname}}, $vals[1]) if (exists $vals[1]);
       }
       $nextisvals=1 if ($nextisheader); $nextisheader=0;
       $nextisheader=1 if ($line=~/METRICS CLASS/);
       $nexthist=1 if ($line=~/normalized_position/);
     } 
  }
  
}


sub runCommand {
	my ($com) = @_;
	if ($com eq ""){
		return "";
    }
    my $error = system(@_);
	if   ($error) { die "Command failed: $error $com\\n"; }
    else          { print "Command successful: $com\\n"; }
}
'''

}

igv_extention_factor = params.BAM_Analysis_Module_RSEM_IGV_BAM2TDF_converter.igv_extention_factor
igv_window_size = params.BAM_Analysis_Module_RSEM_IGV_BAM2TDF_converter.igv_window_size

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 24
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} 
//* platform
//* autofill

process BAM_Analysis_Module_RSEM_IGV_BAM2TDF_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tdf$/) "igvtools_rsem/$filename"}
input:
 val mate from g_347_mate10_g251_131
 set val(name), file(bam) from g250_26_bam_file11_g251_131
 file genomeSizes from g245_54_genomeSizes22_g251_131

output:
 file "*.tdf"  into g251_131_outputFileOut00

when:
(params.run_IGV_TDF_Conversion && (params.run_IGV_TDF_Conversion == "yes")) || !params.run_IGV_TDF_Conversion

script:
pairedText = (params.nucleicAcidType == "dna" && mate == "pair") ? " --pairs " : ""
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
"""
$runSamtools
igvtools count -w ${igv_window_size} -e ${igv_extention_factor} ${pairedText} ${nameFinal} ${name}.tdf ${genomeSizes}
"""
}

//* params.star_index =  ""  //* @input

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 10
    $MEMORY = 50
}
//* platform
if ($HOSTNAME == "hpc.umassmed.edu"){
    $CPU  = 10
    $MEMORY = 20
}
//* platform
//* autofill

process STAR_Module_Map_STAR {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${name}Log.out$/) "star/$filename"}
input:
 val mate from g_347_mate10_g264_31
 set val(name), file(reads) from g256_46_reads01_g264_31
 file star_index from g264_26_starIndex02_g264_31

output:
 set val(name), file("${name}Log.final.out")  into g264_31_outputFileOut00_g264_18
 file "${name}Log.out"  into g264_31_logOut11
 set val(name), file("${name}.bam")  into g264_31_mapped_reads20_g264_30
 set val(name), file("${name}SJ.out.tab")  into g264_31_outputFileTab33
 file "${name}Log.progress.out"  into g264_31_progressOut44
 set val(name), file("${name}Aligned.toTranscriptome.out.bam") optional true  into g264_31_transcriptome_bam50_g264_15
 file "${name}Log.final.out"  into g264_31_logFinalOut62_g_177

container "quay.io/viascientific/rsem:1.0"

when:
(params.run_STAR && (params.run_STAR == "yes")) || !params.run_STAR

script:
params_STAR = params.STAR_Module_Map_STAR.params_STAR
transcriptomeSAM = ""
if (params.run_Salmon_after_STAR && params.run_Salmon_after_STAR == "yes" && params_STAR.indexOf("--quantMode") < 0){
	transcriptomeSAM = " --quantMode TranscriptomeSAM "
}

"""
STAR --runThreadN ${task.cpus} ${params_STAR} ${transcriptomeSAM} --genomeDir ${star_index} --readFilesCommand zcat --readFilesIn $reads --outFileNamePrefix ${name}
echo "Alignment completed."
if [ ! -e "${name}Aligned.toTranscriptome.out.bam" -a -e "${name}Aligned.toTranscriptome.out.sam" ] ; then
    samtools view -S -b ${name}Aligned.toTranscriptome.out.sam > ${name}Aligned.toTranscriptome.out.bam
elif [ ! -e "${name}Aligned.out.bam" -a -e "${name}Aligned.out.sam" ] ; then
    samtools view -S -b ${name}Aligned.out.sam > ${name}Aligned.out.bam
fi
rm -rf *.sam
if [ -e "${name}Aligned.sortedByCoord.out.bam" ] ; then
    mv ${name}Aligned.sortedByCoord.out.bam ${name}.bam
elif [ -e "${name}Aligned.out.bam" ] ; then
    mv ${name}Aligned.out.bam ${name}.bam
fi

"""


}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 10
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 2000
    $CPU  = 1
    $MEMORY = 8
    $QUEUE = "long"
}
//* platform
//* autofill

process STAR_Module_Merge_Bam_and_create_sense_antisense {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_sorted.bam.bai$/) "star/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_sorted.bam$/) "star/$filename"}
input:
 set val(oldname), file(bamfiles) from g264_31_mapped_reads20_g264_30.groupTuple()
 val mate from g_347_mate11_g264_30

output:
 file "*_sorted.bam.bai"  into g264_30_bam_index00
 set val(oldname), file("*_sorted.bam")  into g264_30_bamFile10_g276_1, g264_30_bamFile11_g253_131, g264_30_bamFile10_g253_121, g264_30_bamFile10_g253_143

errorStrategy 'retry'
maxRetries 2

shell:
'''
num=$(echo "!{bamfiles.join(" ")}" | awk -F" " '{print NF-1}')
if [ "${num}" -gt 0 ]; then
    samtools merge !{oldname}.bam !{bamfiles.join(" ")} && samtools sort -o !{oldname}_sorted.bam !{oldname}.bam && samtools index !{oldname}_sorted.bam
else
    mv !{bamfiles.join(" ")} !{oldname}.bam 2>/dev/null || true
    samtools sort  -o !{oldname}_sorted.bam !{oldname}.bam && samtools index !{oldname}_sorted.bam
fi


'''
}


process BAM_Analysis_Module_STAR_bam_sort_index {

input:
 set val(name), file(bam) from g264_30_bamFile10_g253_143

output:
 set val(name), file("bam/*.bam"), file("bam/*.bam.bai")  into g253_143_bam_bai00_g253_134, g253_143_bam_bai00_g253_142

when:
params.run_BigWig_Conversion == "yes" || params.run_RSeQC == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}

"""
$runSamtools
mkdir -p bam
mv ${name}_sorted.bam ${name}_sorted.bam.bai bam/.
"""
}



//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 30
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_STAR_UCSC_BAM2BigWig_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.bw$/) "bigwig_star/$filename"}
input:
 set val(name), file(bam), file(bai) from g253_143_bam_bai00_g253_142
 file genomeSizes from g245_54_genomeSizes21_g253_142

output:
 file "*.bw" optional true  into g253_142_outputFileBw00
 file "publish/*.bw" optional true  into g253_142_publishBw10_g253_145

container 'quay.io/biocontainers/deeptools:3.5.4--pyhdfd78af_1'

when:
params.run_BigWig_Conversion == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = ""
    nameFinal = nameAll
} else {
    runSamtools = "mv $bam ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
deeptools_parameters = params.BAM_Analysis_Module_STAR_UCSC_BAM2BigWig_converter.deeptools_parameters
visualize_bigwig_in_reports = params.BAM_Analysis_Module_STAR_UCSC_BAM2BigWig_converter.visualize_bigwig_in_reports

"""
$runSamtools
bamCoverage ${deeptools_parameters}  -b ${nameFinal} -o ${name}.bw 

if [ "${visualize_bigwig_in_reports}" == "yes" ]; then
	mkdir -p publish
	mv ${name}.bw publish/.
fi
"""

}



process BAM_Analysis_Module_STAR_Genome_Browser {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "genome_browser_star/$filename"}
input:
 file bigwigs from g253_142_publishBw10_g253_145.collect()
 file group_file from g_295_1_g253_145

output:
 file "*"  into g253_145_bigWig_file00

container 'quay.io/viascientific/python-basics:2.0'


script:
try {
    myVariable = group_file
} catch (MissingPropertyException e) {
    group_file = ""
}

genome_build_short= ""
if (params.genome_build == "mousetest_mm10"){
    genome_build_short= "mm10"
} else if (params.genome_build == "human_hg19_refseq"){
    genome_build_short = "hg19"
} else if (params.genome_build == "human_hg38_gencode_v28"){
    genome_build_short = "hg38"
} else if (params.genome_build == "human_hg38_gencode_v34"){
    genome_build_short = "hg38"
} else if (params.genome_build == "mouse_mm10_refseq"){
    genome_build_short = "mm10"
} else if (params.genome_build == "mouse_mm10_gencode_m25"){
    genome_build_short = "mm10"
} else if (params.genome_build == "rat_rn6_refseq"){
    genome_build_short = "rn6"
} else if (params.genome_build == "rat_rn6_ensembl_v86"){
    genome_build_short = "rn6"
} else if (params.genome_build == "zebrafish_GRCz11_ensembl_v95"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_refseq"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_v4.3.2"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "c_elegans_ce11_ensembl_ws245"){
    genome_build_short = "ce11"
} else if (params.genome_build == "d_melanogaster_dm6_refseq"){
    genome_build_short = "dm6"
} else if (params.genome_build == "s_cerevisiae_sacCer3_refseq"){
    genome_build_short = "sacCer3"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v31"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v51"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "e_coli_ASM584v2_refseq"){
    genome_build_short = "ASM584v2"
} else if (params.genome_build == "dog_canFam3_refseq"){
    genome_build_short = "canFam3"
} 

"""

#!/usr/bin/env python

import glob,json,os,sys,csv,random,shutil,requests,subprocess

def check_url_existence(url):
    try:
        response = requests.head(url, allow_redirects=True)
        # Check if the status code is in the success range (200-399)
        return 200 <= response.status_code < 400
    except requests.ConnectionError:
        return False

def get_lib_val(str, lib):
    for key, value in lib.items():
        if key.lower() in str.lower():
            return value
    return None

def find_and_move_folders_with_bw_files(start_dir):
    bigwig_dir = "bigwigs"
    if os.path.exists(bigwig_dir)!=True:
        os.makedirs(bigwig_dir)
    subprocess.getoutput("cp ${bigwigs} bigwigs/. ")

def Generate_HubFile(groupfile):
    # hub.txt
    Hub = open("hub.txt", "w")
    Hub.write("hub UCSCHub \\n")
    Hub.write("shortLabel UCSCHub \\n")
    Hub.write("longLabel UCSCHub \\n")
    Hub.write("genomesFile genomes.txt \\n")
    Hub.write("email support@viascientific.com \\n")
    Hub.write("\\n")
    Hub.close()

    # genomes.txt
    genomes = open("genomes.txt", "w")
    genomes.write("genome ${genome_build_short} \\n")
    genomes.write("trackDb bigwigs/trackDb.txt \\n")
    genomes.write("\\n")
    genomes.close()
    #trackDb = open("bigwigs/trackDb.txt", "w")
    path = r'bigwigs/*.bw'
    files = glob.glob(path)
    files.sort()
    sample={}
    for i in files:
        temp=i.split('.')[0]
        temp=temp.replace('bigwigs/','')
        if temp in sample.keys():
            sample[temp].append(i.replace('bigwigs/',''))
        else:
            sample[temp]=[]
            sample[temp].append(i.replace('bigwigs/',''))
    second_layer_indent=" "*2
    third_layer_indent = " " * 14
    if groupfile == "" or os.stat(groupfile).st_size == 0:
        #No GroupFile
        trackDb = open("bigwigs/trackDb.txt", "w")
        for i in sample.keys():
            trackDb.write('track %s\\n' %i)
            trackDb.write('shortLabel %s\\n' %i)
            trackDb.write('longLabel %s\\n' %i)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for j in sample[i]:
                trackDb.write(second_layer_indent+'track %s\\n' % j)
                trackDb.write(second_layer_indent+'bigDataUrl %s\\n' % j)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %i)
                trackDb.write('\\n')
        trackDb.close()
    else:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_to_sample={}
        for cond in condition:
            if cond not in condition_to_sample.keys():
                condition_to_sample[cond]=[]
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
            else:
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
        trackDb = open("bigwigs/trackDb.txt", "w")
        for cond in condition:
            trackDb.write('track %s\\n' %cond)
            trackDb.write('shortLabel %s\\n' %cond)
            trackDb.write('longLabel %s\\n' %cond)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for samp in condition_to_sample[cond]:
                trackDb.write(second_layer_indent+'track %s\\n' % samp)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'view Signal \\n')
                trackDb.write(second_layer_indent+'visibility full \\n')
                trackDb.write(second_layer_indent+'viewLimits 0:20 \\n')
                trackDb.write(second_layer_indent+'autoScale on \\n')
                trackDb.write(second_layer_indent+'maxHeightPixels 128:20:8 \\n')
                trackDb.write(second_layer_indent+'configurable on \\n')
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %cond)
                trackDb.write('\\n')
                for file in sample[samp]:
                    trackDb.write(third_layer_indent + 'track %s\\n' % file)
                    trackDb.write(third_layer_indent + 'bigDataUrl %s\\n' % file)
                    trackDb.write(third_layer_indent + 'shortLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'longLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'type bigWig\\n')
                    trackDb.write(third_layer_indent+'parent %s\\n' %samp)

                    trackDb.write('\\n')

        trackDb.close()


def Generating_Json_files(groupfile):
    publishWebDir = '{{DNEXT_WEB_REPORT_DIR}}/' + 'genome_browser_star' + "/bigwigs"
    locusLib = {}
    # MYC locations
    locusLib["hg19"] = "chr8:128,746,315-128,755,680"
    locusLib["hg38"] = "chr8:127,733,434-127,744,951"
    locusLib["mm10"] = "chr15:61,983,341-61,992,361"
    locusLib["rn6"] = "chr7:102,584,313-102,593,240"
    locusLib["dm6"] = "chrX:3,371,159-3,393,697"
    locusLib["canFam3"] = "chr13:25,198,772-25,207,309"

    cytobandLib = {}
    cytobandLib["hg19"] = "https://igv-genepattern-org.s3.amazonaws.com/genomes/seq/hg19/cytoBand.txt"
    cytobandLib["hg38"] = "https://s3.amazonaws.com/igv.org.genomes/hg38/annotations/cytoBandIdeo.txt.gz"
    cytobandLib["mm10"] = "https://s3.amazonaws.com/igv.broadinstitute.org/annotations/mm10/cytoBandIdeo.txt.gz"
    cytobandLib["rn6"] = "https://s3.amazonaws.com/igv.org.genomes/rn6/cytoBand.txt.gz"
    cytobandLib["dm6"] = "https://s3.amazonaws.com/igv.org.genomes/dm6/cytoBandIdeo.txt.gz"
    cytobandLib["ce11"] = "https://s3.amazonaws.com/igv.org.genomes/ce11/cytoBandIdeo.txt.gz"
    cytobandLib["canFam3"] = "https://s3.amazonaws.com/igv.org.genomes/canFam3/cytoBandIdeo.txt.gz"

    # Get the basename of the original path
    gtf_source_base_name = os.path.basename("${params.gtf_source}")
    gtf_source_sorted = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz")
    print(gtf_source_sorted)
    gtf_source_sorted_index = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz.tbi")
    print(gtf_source_sorted_index)


    data = {}
    data["reference"] = {}
    data["reference"]["id"] = "${params.genome_build}"
    data["reference"]["name"] = "${params.genome_build}"
    data["reference"]["fastaURL"] = "${params.genome_source}"
    data["reference"]["indexURL"] = "${params.genome_source}.fai"
    cytobandurl = get_lib_val("${params.genome_build}", cytobandLib)
    locusStr = get_lib_val("${params.genome_build}", locusLib)
    if cytobandurl is not None:
        data["reference"]["cytobandURL"] = cytobandurl
    if locusStr is not None:
        data["locus"] = []
        data["locus"].append(locusStr)
    data["tracks"] = []
    # prepare gtf Track
    gtfTrack = {}
    gtfTrack["name"] = "${params.genome_build}"
    gtfTrack["gtf"] = "gtf"
    if check_url_existence(gtf_source_sorted):
        gtfTrack["url"] = gtf_source_sorted
    else:
        gtfTrack["url"] = "${params.gtf_source}"
    if check_url_existence(gtf_source_sorted_index):
        gtfTrack["indexURL"] = gtf_source_sorted_index

    # prepare cytobands Track
    if groupfile and os.path.isfile(groupfile) and os.stat(groupfile).st_size != 0:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_color_dict = dict()
        for cond in condition:
            r = lambda: random.randint(0, 255)
            condition_color_dict[cond] = '#%02X%02X%02X' % (r(), r(), r())

    # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                if basename in data_parsed:
                    newdata[file] = data_parsed[basename]
                    newdata[file]["fullname"] = file

        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": condition_color_dict[newdata[j]['group']],
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()
    else:

        # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                newdata[file] = basename
                newdata[file] = file
        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            r = lambda: random.randint(0, 255)
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": '#%02X%02X%02X' % (r(), r(), r()),
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()

if __name__ == "__main__":
    find_and_move_folders_with_bw_files(".")
    Generate_HubFile(groupfile="${group_file}")
    Generating_Json_files(groupfile="${group_file}")

"""
}

//* params.bed =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 10
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_STAR_RSeQC {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "rseqc_star/$filename"}
input:
 set val(name), file(bam), file(bai) from g253_143_bam_bai00_g253_134
 file bed from g245_54_bed31_g253_134
 val mate from g_347_mate12_g253_134

output:
 file "*"  into g253_134_outputFileOut08_g_177

container 'quay.io/viascientific/rseqc:1.0'

when:
(params.run_RSeQC && (params.run_RSeQC == "yes")) || !params.run_RSeQC

script:
run_bam_stat = params.BAM_Analysis_Module_STAR_RSeQC.run_bam_stat
run_read_distribution = params.BAM_Analysis_Module_STAR_RSeQC.run_read_distribution
run_inner_distance = params.BAM_Analysis_Module_STAR_RSeQC.run_inner_distance
run_junction_annotation = params.BAM_Analysis_Module_STAR_RSeQC.run_junction_annotation
run_junction_saturation = params.BAM_Analysis_Module_STAR_RSeQC.run_junction_saturation
//run_geneBody_coverage and run_infer_experiment needs subsampling
run_geneBody_coverage = params.BAM_Analysis_Module_STAR_RSeQC.run_geneBody_coverage
run_infer_experiment = params.BAM_Analysis_Module_STAR_RSeQC.run_infer_experiment
"""
if [ "$run_bam_stat" == "true" ]; then bam_stat.py  -i ${bam} > ${name}.bam_stat.txt; fi
if [ "$run_read_distribution" == "true" ]; then read_distribution.py  -i ${bam} -r ${bed}> ${name}.read_distribution.out; fi


if [ "$run_infer_experiment" == "true" -o "$run_geneBody_coverage" == "true" ]; then
	numAlignedReads=\$(samtools view -c -F 4 $bam)

	if [ "\$numAlignedReads" -gt 1000000 ]; then
    	echo "Read number is greater than 1000000. Subsampling..."
    	finalRead=1000000
    	fraction=\$(samtools idxstats  $bam | cut -f3 | awk -v ct=\$finalRead 'BEGIN {total=0} {total += \$1} END {print ct/total}')
    	samtools view -b -s \${fraction} $bam > ${name}_sampled.bam
    	samtools index ${name}_sampled.bam
    	if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i ${name}_sampled.bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i ${name}_sampled.bam  -r $bed -o ${name}; fi
	else
		if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i $bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i $bam  -r $bed -o ${name}; fi
	fi

fi


if [ "${mate}" == "pair" ]; then
	if [ "$run_inner_distance" == "true" ]; then inner_distance.py -i $bam  -r $bed -o ${name}.inner_distance > stdout.txt; fi
	if [ "$run_inner_distance" == "true" ]; then head -n 2 stdout.txt > ${name}.inner_distance_mean.txt; fi
fi
if [ "$run_junction_annotation" == "true" ]; then junction_annotation.py -i $bam  -r $bed -o ${name}.junction_annotation 2> ${name}.junction_annotation.log; fi
if [ "$run_junction_saturation" == "true" ]; then junction_saturation.py -i $bam  -r $bed -o ${name}; fi
if [ -e class.log ] ; then mv class.log ${name}_class.log; fi
if [ -e log.txt ] ; then mv log.txt ${name}_log.txt; fi
if [ -e stdout.txt ] ; then mv stdout.txt ${name}_stdout.txt; fi


"""

}

//* params.pdfbox_path =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 32
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 240
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "short"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 500
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
}
//* platform
//* autofill

process BAM_Analysis_Module_STAR_Picard {

input:
 set val(name), file(bam) from g264_30_bamFile10_g253_121

output:
 file "*_metrics"  into g253_121_outputFileOut00_g253_82
 file "results/*.pdf"  into g253_121_outputFilePdf12_g253_82

container 'quay.io/viascientific/picard:1.0'

when:
(params.run_Picard_CollectMultipleMetrics && (params.run_Picard_CollectMultipleMetrics == "yes")) || !params.run_Picard_CollectMultipleMetrics

script:
"""
picard CollectMultipleMetrics OUTPUT=${name}_multiple.out VALIDATION_STRINGENCY=LENIENT INPUT=${bam}
mkdir results && java -jar ${params.pdfbox_path} PDFMerger *.pdf results/${name}_multi_metrics.pdf
"""
}


process BAM_Analysis_Module_STAR_Picard_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "picard_summary_star/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "rseqc_summary_star/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /results\/.*.pdf$/) "picard_summary_pdf_star/$filename"}
input:
 file picardOut from g253_121_outputFileOut00_g253_82.collect()
 val mate from g_347_mate11_g253_82
 file picardPdf from g253_121_outputFilePdf12_g253_82.collect()

output:
 file "*.tsv"  into g253_82_outputFileTSV00
 file "results/*.pdf"  into g253_82_outputFilePdf11

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage; 
use Data::Dumper;

runCommand("mkdir results && mv *.pdf results/. ");

my $indir = $ENV{'PWD'};
my $outd = $ENV{'PWD'};
my @files = ();
my @outtypes = ("CollectRnaSeqMetrics", "alignment_summary_metrics", "base_distribution_by_cycle_metrics", "insert_size_metrics", "quality_by_cycle_metrics", "quality_distribution_metrics" );

foreach my $outtype (@outtypes)
{
my $ext="_multiple.out";
$ext.=".$outtype" if ($outtype ne "CollectRnaSeqMetrics");
@files = <$indir/*$ext>;

my @rowheaders=();
my @libs=();
my %metricvals=();
my %histvals=();

my $pdffile="";
my $libname="";
foreach my $d (@files){
  my $libname=basename($d, $ext);
  print $libname."\\n";
  push(@libs, $libname); 
  getMetricVals($d, $libname, \\%metricvals, \\%histvals, \\@rowheaders);
}

my $sizemetrics = keys %metricvals;
write_results("$outd/$outtype.stats.tsv", \\@libs,\\%metricvals, \\@rowheaders, "metric") if ($sizemetrics>0);
my $sizehist = keys %histvals;
write_results("$outd/$outtype.hist.tsv", \\@libs,\\%histvals, "none", "nt") if ($sizehist>0);

}

sub write_results
{
  my ($outfile, $libs, $vals, $rowheaders, $name )=@_;
  open(OUT, ">$outfile");
  print OUT "$name\\t".join("\\t", @{$libs})."\\n";
  my $size=0;
  $size=scalar(@{${$vals}{${$libs}[0]}}) if(exists ${$libs}[0] and exists ${$vals}{${$libs}[0]} );
  
  for (my $i=0; $i<$size;$i++)
  { 
    my $rowname=$i;
    $rowname = ${$rowheaders}[$i] if ($name=~/metric/);
    print OUT $rowname;
    foreach my $lib (@{$libs})
    {
      print OUT "\\t".${${$vals}{$lib}}[$i];
    } 
    print OUT "\\n";
  }
  close(OUT);
}

sub getMetricVals{
  my ($filename, $libname, $metricvals, $histvals,$rowheaders)=@_;
  if (-e $filename){
     my $nextisheader=0;
     my $nextisvals=0;
     my $nexthist=0;
     open(IN, $filename);
     while(my $line=<IN>)
     {
       chomp($line);
       @{$rowheaders}=split(/\\t/, $line) if ($nextisheader && !scalar(@{$rowheaders})); 
       if ($nextisvals) {
         @{${$metricvals}{$libname}}=split(/\\t/, $line);
         $nextisvals=0;
       }
       if($nexthist){
          my @vals=split(/[\\s\\t]+/,$line); 
          push(@{${$histvals}{$libname}}, $vals[1]) if (exists $vals[1]);
       }
       $nextisvals=1 if ($nextisheader); $nextisheader=0;
       $nextisheader=1 if ($line=~/METRICS CLASS/);
       $nexthist=1 if ($line=~/normalized_position/);
     } 
  }
  
}


sub runCommand {
	my ($com) = @_;
	if ($com eq ""){
		return "";
    }
    my $error = system(@_);
	if   ($error) { die "Command failed: $error $com\\n"; }
    else          { print "Command successful: $com\\n"; }
}
'''

}

igv_extention_factor = params.BAM_Analysis_Module_STAR_IGV_BAM2TDF_converter.igv_extention_factor
igv_window_size = params.BAM_Analysis_Module_STAR_IGV_BAM2TDF_converter.igv_window_size

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 24
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} 
//* platform
//* autofill

process BAM_Analysis_Module_STAR_IGV_BAM2TDF_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tdf$/) "igvtools_star/$filename"}
input:
 val mate from g_347_mate10_g253_131
 set val(name), file(bam) from g264_30_bamFile11_g253_131
 file genomeSizes from g245_54_genomeSizes22_g253_131

output:
 file "*.tdf"  into g253_131_outputFileOut00

when:
(params.run_IGV_TDF_Conversion && (params.run_IGV_TDF_Conversion == "yes")) || !params.run_IGV_TDF_Conversion

script:
pairedText = (params.nucleicAcidType == "dna" && mate == "pair") ? " --pairs " : ""
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
"""
$runSamtools
igvtools count -w ${igv_window_size} -e ${igv_extention_factor} ${pairedText} ${nameFinal} ${name}.tdf ${genomeSizes}
"""
}

//* params.gtf =  ""  //* @input


process Bam_Quantify_Module_STAR_featureCounts {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "featureCounts_after_STAR/$filename"}
input:
 set val(name), file(bam) from g264_30_bamFile10_g276_1
 val paired from g_347_mate11_g276_1
 each run_params from g276_0_run_parameters02_g276_1
 file gtf from g245_54_gtfFile03_g276_1

output:
 file "*"  into g276_1_outputFileTSV00_g276_2

script:
pairText = ""
if (paired == "pair"){
    pairText = "-p"
}

run_name = run_params["run_name"] 
run_parameters = run_params["run_parameters"] 

"""
featureCounts ${pairText} ${run_parameters} -a ${gtf} -o ${name}@${run_name}@fCounts.txt ${bam}
## remove first line
sed -i '1d' ${name}@${run_name}@fCounts.txt

"""
}

//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 30
    $CPU  = 1
    $MEMORY = 10
    $QUEUE = "short"
}
//* platform
//* autofill

process Bam_Quantify_Module_STAR_featureCounts_summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_featureCounts.tsv$/) "featureCounts_after_STAR_summary/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_featureCounts.sum.tsv$/) "featureCounts_after_STAR_details/$filename"}
input:
 file featureCountsOut from g276_1_outputFileTSV00_g276_2.collect()

output:
 file "*_featureCounts.tsv"  into g276_2_outputFile00_g305_25, g276_2_outputFile00_g305_24
 file "*_featureCounts.sum.tsv"  into g276_2_outFileTSV11

shell:
'''
#!/usr/bin/env perl

# Step 1: Merge count files
my %tf = ( expected_count => 6 );
my @run_name=();
chomp(my $contents = `ls *@fCounts.txt`);
my @files = split(/[\\n]+/, $contents);
foreach my $file (@files){
    $file=~/(.*)\\@(.*)\\@fCounts\\.txt/;
    my $runname = $2;
    push(@run_name, $runname) unless grep{$_ eq $runname} @run_name;
}


my @expectedCount_ar = ("expected_count");
for($l = 0; $l <= $#run_name; $l++) {
    my $runName = $run_name[$l];
    for($ll = 0; $ll <= $#expectedCount_ar; $ll++) {
        my $expectedCount = $expectedCount_ar[$ll];
    
        my @a=();
        my %b=();
        my %c=();
        my $i=0;
        chomp(my $contents = `ls *\\@${runName}\\@fCounts.txt`);
        my @files = split(/[\\n]+/, $contents);
        foreach my $file (@files){
        $i++;
        $file=~/(.*)\\@${runName}\\@fCounts\\.txt/;
        my $libname = $1; 
        $a[$i]=$libname;
        open IN, $file;
            $_=<IN>;
            while(<IN>){
                my @v=split; 
                $b{$v[0]}{$i}=$v[$tf{$expectedCount}];
                $c{$v[0]}=$v[5]; #length column
            }
            close IN;
        }
        my $outfile="$runName"."_featureCounts.tsv";
        open OUT, ">$outfile";
        if ($runName eq "transcript_id") {
            print OUT "transcript\tlength";
        } else {
            print OUT "gene\tlength";
        }
    
        for(my $j=1;$j<=$i;$j++) {
            print OUT "\t$a[$j]";
        }
        print OUT "\n";
    
        foreach my $key (keys %b) {
            print OUT "$key\t$c{$key}";
            for(my $j=1;$j<=$i;$j++){
                print OUT "\t$b{$key}{$j}";
            }
            print OUT "\n";
        }
        close OUT;
         
    }
}


	

# Step 2: Merge summary files
for($l = 0; $l <= $#run_name; $l++) {
    my $runName = $run_name[$l];
    my @a=();
    my %b=();
    my $i=0;
    chomp(my $contents = `ls *\\@${runName}\\@fCounts.txt.summary`);
    my @files = split(/[\\n]+/, $contents);
    foreach my $file (@files){
        $i++;
        $file=~/(.*)\\@${runName}\\@fCounts\\.txt\\.summary/;
        my $libname = $1; 
        $a[$i]=$libname;
        open IN, $file;
        $_=<IN>;
        while(<IN>){
            my @v=split; 
            $b{$v[0]}{$i}=$v[1];
        }
        close IN;
    }
    my $outfile="$runName"."_featureCounts.sum.tsv";
    open OUT, ">$outfile";
    print OUT "criteria";
    for(my $j=1;$j<=$i;$j++) {
        print OUT "\t$a[$j]";
    }
    print OUT "\n";
    
    foreach my $key (keys %b) {
        print OUT "$key";
        for(my $j=1;$j<=$i;$j++){
            print OUT "\t$b{$key}{$j}";
        }
        print OUT "\n";
    }
    close OUT;
}

'''
}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_featurecounts_Prepare_DESeq2 {

input:
 file counts from g276_2_outputFile00_g305_24
 file groups_file from g_295_1_g305_24
 file compare_file from g_294_2_g305_24
 val run_DESeq2 from g_308_3_g305_24

output:
 file "DE_reports"  into g305_24_outputFile00_g305_37
 val "_des"  into g305_24_postfix10_g305_33
 file "DE_reports/outputs/*_all_deseq2_results.tsv"  into g305_24_outputFile21_g305_33

container 'quay.io/viascientific/de_module:4.0'

when:
run_DESeq2 == 'yes'

script:

feature_type = params.DE_module_STAR_featurecounts_Prepare_DESeq2.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_distribution
include_all2all = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_all2all
include_pca = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_pca

filter_type = params.DE_module_STAR_featurecounts_Prepare_DESeq2.filter_type
min_count = params.DE_module_STAR_featurecounts_Prepare_DESeq2.min_count
min_samples = params.DE_module_STAR_featurecounts_Prepare_DESeq2.min_samples
min_counts_per_sample = params.DE_module_STAR_featurecounts_Prepare_DESeq2.min_counts_per_sample
excluded_events = params.DE_module_STAR_featurecounts_Prepare_DESeq2.excluded_events

include_batch_correction = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_batch_correction
batch_correction_column = params.DE_module_STAR_featurecounts_Prepare_DESeq2.batch_correction_column
batch_correction_group_column = params.DE_module_STAR_featurecounts_Prepare_DESeq2.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_STAR_featurecounts_Prepare_DESeq2.batch_normalization_algorithm

transformation = params.DE_module_STAR_featurecounts_Prepare_DESeq2.transformation
pca_color = params.DE_module_STAR_featurecounts_Prepare_DESeq2.pca_color
pca_shape = params.DE_module_STAR_featurecounts_Prepare_DESeq2.pca_shape
pca_fill = params.DE_module_STAR_featurecounts_Prepare_DESeq2.pca_fill
pca_transparency = params.DE_module_STAR_featurecounts_Prepare_DESeq2.pca_transparency
pca_label = params.DE_module_STAR_featurecounts_Prepare_DESeq2.pca_label

include_deseq2 = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_deseq2
input_mode = params.DE_module_STAR_featurecounts_Prepare_DESeq2.input_mode
design = params.DE_module_STAR_featurecounts_Prepare_DESeq2.design
fitType = params.DE_module_STAR_featurecounts_Prepare_DESeq2.fitType
use_batch_corrected_in_DE = params.DE_module_STAR_featurecounts_Prepare_DESeq2.use_batch_corrected_in_DE
apply_shrinkage = params.DE_module_STAR_featurecounts_Prepare_DESeq2.apply_shrinkage
shrinkage_type = params.DE_module_STAR_featurecounts_Prepare_DESeq2.shrinkage_type
include_volcano = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_volcano
include_ma = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_ma
include_heatmap = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_heatmap

padj_significance_cutoff = params.DE_module_STAR_featurecounts_Prepare_DESeq2.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_STAR_featurecounts_Prepare_DESeq2.fc_significance_cutoff
padj_floor = params.DE_module_STAR_featurecounts_Prepare_DESeq2.padj_floor
fc_ceiling = params.DE_module_STAR_featurecounts_Prepare_DESeq2.fc_ceiling

convert_names = params.DE_module_STAR_featurecounts_Prepare_DESeq2.convert_names
count_file_names = params.DE_module_STAR_featurecounts_Prepare_DESeq2.count_file_names
converted_name = params.DE_module_STAR_featurecounts_Prepare_DESeq2.converted_name
org_db = params.DE_module_STAR_featurecounts_Prepare_DESeq2.org_db
num_labeled = params.DE_module_STAR_featurecounts_Prepare_DESeq2.num_labeled
highlighted_genes = params.DE_module_STAR_featurecounts_Prepare_DESeq2.highlighted_genes
include_volcano_highlighted = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_volcano_highlighted
include_ma_highlighted = params.DE_module_STAR_featurecounts_Prepare_DESeq2.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name, org_db},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm, use_batch_corrected_in_DE},{include_batch_correction="false"},{include_deseq2="true", design, fitType, apply_shrinkage, shrinkage_type, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, org_db, num_labeled, highlighted_genes},{include_deseq2="false"},{apply_shrinkage="true", shrinkage_type},{apply_shrinkage="false"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples, min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{design, fitType, use_batch_corrected_in_DE, apply_shrinkage, shrinkage_type},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name, org_db},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_deseq2 = include_deseq2 == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
apply_shrinkage = apply_shrinkage == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir reports
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_DESeq2.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-DESeq2 ${include_deseq2} --input-mode ${input_mode} --design '${design}' --fitType ${fitType} --use-batch-correction-in-DE ${use_batch_corrected_in_DE} --apply-shrinkage ${apply_shrinkage} --shrinkage-type ${shrinkage_type} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --org-db ${org_db} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_featurecounts_Prepare_LimmaVoom {

input:
 file counts from g276_2_outputFile00_g305_25
 file groups_file from g_295_1_g305_25
 file compare_file from g_294_2_g305_25
 val run_limmaVoom from g_358_3_g305_25

output:
 file "DE_reports"  into g305_25_outputFile00_g305_39
 val "_lv"  into g305_25_postfix10_g305_41
 file "DE_reports/outputs/*_all_limmaVoom_results.tsv"  into g305_25_outputFile21_g305_41

container 'quay.io/viascientific/de_module:4.0'

when:
run_limmaVoom == 'yes'

script:

feature_type = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_distribution
include_all2all = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_all2all
include_pca = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_pca

filter_type = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.filter_type
min_count = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.min_count
min_samples = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.min_samples
min_counts_per_sample = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.min_counts_per_sample
excluded_events = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.excluded_events

include_batch_correction = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_batch_correction
batch_correction_column = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.batch_correction_column
batch_correction_group_column = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.batch_normalization_algorithm

transformation = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.transformation
pca_color = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.pca_color
pca_shape = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.pca_shape
pca_fill = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.pca_fill
pca_transparency = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.pca_transparency
pca_label = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.pca_label

include_limma = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_limma
use_batch_corrected_in_DE = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.use_batch_corrected_in_DE
normalization_method = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.normalization_method
logratioTrim = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.logratioTrim
sumTrim = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.sumTrim
Acutoff = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.Acutoff
doWeighting = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.doWeighting
p = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.p
include_volcano = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_volcano
include_ma = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_ma
include_heatmap = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_heatmap

padj_significance_cutoff = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.fc_significance_cutoff
padj_floor = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.padj_floor
fc_ceiling = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.fc_ceiling

convert_names = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.convert_names
count_file_names = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.count_file_names
converted_name = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.converted_name
num_labeled = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.num_labeled
highlighted_genes = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.highlighted_genes
include_volcano_highlighted = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_volcano_highlighted
include_ma_highlighted = params.DE_module_STAR_featurecounts_Prepare_LimmaVoom.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm,use_batch_corrected_in_DE},{include_batch_correction="false"},{include_limma="true", normalization_method, logratioTrim, sumTrim, doWeighting, Acutoff, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, num_labeled, highlighted_genes},{include_limma="false"},{normalization_method="TMM", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="TMMwsp", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="RLE"},{normalization_method="upperquartile", p},{normalization_method="none"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples,min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_limma, use_batch_corrected_in_DE},{normalization_method,logratioTrim,sumTrim,doWeighting,Acutoff,p},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_limma = include_limma == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

doWeighting = doWeighting == 'true' ? 'TRUE' : 'FALSE'
TMM_args = normalization_method.equals('TMM') || normalization_method.equals('TMMwsp') ? '--logratio-trim ' + logratioTrim + ' --sum-trim ' + sumTrim + ' --do-weighting ' + doWeighting + ' --A-cutoff="' + Acutoff + '"' : ''
upperquartile_args = normalization_method.equals('upperquartile') ? '--p ' + p : ''

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_limmaVoom.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-limma ${include_limma} \
--use-batch-correction-in-DE ${use_batch_corrected_in_DE} --normalization-method ${normalization_method} ${TMM_args} ${upperquartile_args} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}


//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 2000
    $CPU  = 1
    $MEMORY = 8
    $QUEUE = "long"
}
//* platform
//* autofill

process STAR_Module_merge_transcriptome_bam {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_sorted.*bam$/) "star/$filename"}
input:
 set val(oldname), file(bamfiles) from g264_31_transcriptome_bam50_g264_15.groupTuple()

output:
 set val(oldname), file("${oldname}.bam")  into g264_15_merged_bams00
 set val(oldname), file("*_sorted*bai")  into g264_15_bam_index11
 set val(oldname), file("*_sorted*bam")  into g264_15_sorted_bam23_g276_9

shell:
'''
num=$(echo "!{bamfiles.join(" ")}" | awk -F" " '{print NF-1}')
if [ "${num}" -gt 0 ]; then
    samtools merge !{oldname}.bam !{bamfiles.join(" ")} && samtools sort -o !{oldname}_sorted.bam !{oldname}.bam && samtools index !{oldname}_sorted.bam
else
    mv !{bamfiles.join(" ")} !{oldname}.bam 2>/dev/null || true
    samtools sort  -o !{oldname}_sorted.bam !{oldname}.bam && samtools index !{oldname}_sorted.bam
fi
'''
}

//* params.salmon_index =  ""  //* @input
//* params.genome_sizes =  ""  //* @input
//* params.gtf =  ""  //* @input
//* @style @multicolumn:{fragment_length,standard_deviation} @condition:{single_or_paired_end_reads="single", fragment_length,standard_deviation}, {single_or_paired_end_reads="pair"}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 4
    $MEMORY = 20 
}
//* platform
//* platform
//* autofill


process Bam_Quantify_Module_STAR_salmon_bam_quant {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /salmon_${name}$/) "salmon_bam_count_star/$filename"}
input:
 val mate from g_347_mate10_g276_9
 file gtf from g245_54_gtfFile01_g276_9
 file genome from g245_54_genome12_g276_9
 set val(name), file(bam) from g264_15_sorted_bam23_g276_9
 val runSalmonBamCount from g_277_4_g276_9

output:
 file "salmon_${name}"  into g276_9_outputDir00_g276_14, g276_9_outputDir017_g_177

container 'quay.io/viascientific/salmon:1.0'

when:
runSalmonBamCount == "yes"

script:
salmon_parameters = params.Bam_Quantify_Module_STAR_salmon_bam_quant.salmon_parameters
libType = params.Bam_Quantify_Module_STAR_salmon_bam_quant.libType
//. strandedness = meta.single_end ? 'U' : 'IU'
//     if (meta.strandedness == 'forward') {
//         strandedness = meta.single_end ? 'SF' : 'ISF'
//     } else if (meta.strandedness == 'reverse') {
//         strandedness = meta.single_end ? 'SR' : 'ISR'
"""
# filter_gtf_for_genes_in_genome.py --gtf ${gtf} --fasta ${genome} -o genome_filtered_genes.gtf
gffread -F -w transcripts_raw.fa -g ${genome} ${gtf}
cut -d ' ' -f1 transcripts_raw.fa > transcripts.fa
salmon quant -t transcripts.fa --threads $task.cpus --libType=$libType -a $bam  $salmon_parameters  -o salmon_${name} 



if [ -f salmon_${name}/quant.sf ]; then
  mv salmon_${name}/quant.sf  salmon_${name}/abundance_isoforms.tsv
fi


"""

}

//* params.gtf =  ""  //* @input

//* autofill
//* platform
//* platform
//* autofill

process Bam_Quantify_Module_STAR_Salmon_transcript_to_gene_count {

input:
 file outDir from g276_9_outputDir00_g276_14
 file gtf from g245_54_gtfFile01_g276_14

output:
 file newoutDir  into g276_14_outputDir00_g276_15

shell:
newoutDir = "genes_" + outDir
'''
#!/usr/bin/env perl
use strict;
use Getopt::Long;
use IO::File;
use Data::Dumper;

my $gtf_file = "!{gtf}";
my $transcript_matrix_in = "!{outDir}/abundance_isoforms.tsv";
my $transcript_matrix_out = "!{outDir}/abundance_genes.tsv";
open(IN, "<$gtf_file") or die "Can't open $gtf_file.\\n";
my %all_genes; # save gene_id of transcript_id
while(<IN>){
  next if(/^##/); #ignore header
  chomp;
  my %attribs = ();
  my ($chr, $source, $type, $start, $end, $score,
    $strand, $phase, $attributes) = split("\\t");
  my @add_attributes = split(";", $attributes);
  # store ids and additional information in second hash
  foreach my $attr ( @add_attributes ) {
     next unless $attr =~ /^\\s*(.+)\\s(.+)$/;
     my $c_type  = $1;
     my $c_value = $2;
     $c_value =~ s/\\"//g;
     if($c_type  && $c_value){
       if(!exists($attribs{$c_type})){
         $attribs{$c_type} = [];
       }
       push(@{ $attribs{$c_type} }, $c_value);
     }
  }
  #work with the information from the two hashes...
  if(exists($attribs{'transcript_id'}->[0]) && exists($attribs{'gene_id'}->[0])){
    if(!exists($all_genes{$attribs{'transcript_id'}->[0]})){
        $all_genes{$attribs{'transcript_id'}->[0]} = $attribs{'gene_id'}->[0];
    }
  } 
}


# print Dumper \\%all_genes;

#Parse the salmon input file, determine gene IDs for each transcript, and calculate sum TPM values
my %gene_exp;
my %gene_length;
my %samples;
my $ki_fh = IO::File->new($transcript_matrix_in, 'r');
my $header = '';
my $h = 0;
while (my $ki_line = $ki_fh->getline) {
  $h++;
  chomp($ki_line);
  my @ki_entry = split("\\t", $ki_line);
  my $s = 0;
  if ($h == 1){
    $header = $ki_line;
    my $first_col = shift @ki_entry;
    my $second_col = shift @ki_entry;
    foreach my $sample (@ki_entry){
      $s++;
      $samples{$s}{name} = $sample;
    }
    next;
  }
  my $trans_id = shift @ki_entry;
  my $length = shift @ki_entry;
  my $gene_id;
  if ($all_genes{$trans_id}){
    $gene_id = $all_genes{$trans_id};
  }elsif($trans_id =~ /ERCC/){
    $gene_id = $trans_id;
  }else{
    print "\\n\\nCould not identify gene id from trans id: $trans_id\\n\\n";
  }

  $s = 0;
  foreach my $value (@ki_entry){
    $s++;
    $gene_exp{$gene_id}{$s} += $value;
  }
  if ($gene_length{$gene_id}){
    $gene_length{$gene_id} = $length if ($length > $gene_length{$gene_id});
  }else{
    $gene_length{$gene_id} = $length;
  }

}
$ki_fh->close;

my $ko_fh = IO::File->new($transcript_matrix_out, 'w');
unless ($ko_fh) { die('Failed to open file: '. $transcript_matrix_out); }

print $ko_fh "$header\\n";
foreach my $gene_id (sort {$a cmp $b} keys %gene_exp){
  print $ko_fh "$gene_id\\t$gene_length{$gene_id}\\t";
  my @vals;
  foreach my $s (sort {$a <=> $b} keys %samples){
     push(@vals, $gene_exp{$gene_id}{$s});
  }
  my $val_string = join("\\t", @vals);
  print $ko_fh "$val_string\\n";
}


$ko_fh->close;
if (checkFile("!{outDir}")){
	rename ("!{outDir}", "!{newoutDir}");
}

sub checkFile {
    my ($file) = @_;
    print "$file\\n";
    return 1 if ( -e $file );
    return 0;
}

'''
}

//* params.gtf =  ""  //* @input

//* autofill
//* platform
//* platform
//* autofill

process Bam_Quantify_Module_STAR_Salmon_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "salmon_bam_count_star_summary/$filename"}
input:
 file salmonOut from g276_14_outputDir00_g276_15.collect()
 file gtf from g245_54_gtfFile01_g276_15

output:
 file "*.tsv"  into g276_15_outputFile00_g302_25, g276_15_outputFile00_g302_24

shell:
'''
#!/usr/bin/env perl
use Data::Dumper;
use strict;

### Parse gtf file
my $gtf_file = "!{gtf}";
open(IN, "<$gtf_file") or die "Can't open $gtf_file.\\n";
my %all_genes; # save gene_id of transcript_id
my %all_trans; # map transcript_id of genes
while(<IN>){
  next if(/^##/); #ignore header
  chomp;
  my %attribs = ();
  my ($chr, $source, $type, $start, $end, $score,
    $strand, $phase, $attributes) = split("\\t");
  my @add_attributes = split(";", $attributes);
  # store ids and additional information in second hash
  foreach my $attr ( @add_attributes ) {
     next unless $attr =~ /^\\s*(.+)\\s(.+)$/;
     my $c_type  = $1;
     my $c_value = $2;
     $c_value =~ s/\\"//g;
     if($c_type  && $c_value){
       if(!exists($attribs{$c_type})){
         $attribs{$c_type} = [];
       }
       push(@{ $attribs{$c_type} }, $c_value);
     }
  }
  #work with the information from the two hashes...
  if(exists($attribs{'transcript_id'}->[0]) && exists($attribs{'gene_id'}->[0])){
    if(!exists($all_genes{$attribs{'transcript_id'}->[0]})){
        $all_genes{$attribs{'transcript_id'}->[0]} = $attribs{'gene_id'}->[0];
    }
    if(!exists($all_trans{$attribs{'gene_id'}->[0]})){
        $all_trans{$attribs{'gene_id'}->[0]} = $attribs{'transcript_id'}->[0];
    } else {
    	if (index($all_trans{$attribs{'gene_id'}->[0]}, $attribs{'transcript_id'}->[0]) == -1) {
			$all_trans{$attribs{'gene_id'}->[0]} = $all_trans{$attribs{'gene_id'}->[0]} . "," .$attribs{'transcript_id'}->[0];
		}
    	
    }
  } 
}


print Dumper \\%all_trans;



#### Create summary table

my %tf = (
        expected_count => 4,
        tpm => 3
    );

my $indir = $ENV{'PWD'};
my $outdir = $ENV{'PWD'};

my @gene_iso_ar = ("genes","isoforms");
my @tpm_fpkm_expectedCount_ar = ("expected_count", "tpm");
for(my $l = 0; $l <= $#gene_iso_ar; $l++) {
    my $gene_iso = $gene_iso_ar[$l];
    for(my $ll = 0; $ll <= $#tpm_fpkm_expectedCount_ar; $ll++) {
        my $tpm_fpkm_expectedCount = $tpm_fpkm_expectedCount_ar[$ll];

        opendir D, $indir or die "Could not open $indir\\n";
        my @alndirs = sort { $a cmp $b } grep /^genes_salmon_/, readdir(D);
        closedir D;
    
        my @a=();
        my %b=();
        my %c=();
        my $i=0;
        foreach my $d (@alndirs){ 
            my $dir = "${indir}/$d";
            print $d."\\n";
            my $libname=$d;
            $libname=~s/genes_salmon_//;
            $i++;
            $a[$i]=$libname;
            open IN,"${dir}/abundance_${gene_iso}.tsv";
            $_=<IN>;
            while(<IN>)
            {
                my @v=split; 
                # $v[0] -> transcript_id
                # $all_genes{$v[0]} -> $gene_id
                if ($gene_iso eq "isoforms"){
                	$c{$v[0]}=$all_genes{$v[0]};
                } elsif ($gene_iso eq "genes"){
                	$c{$v[0]}=$all_trans{$v[0]};
                } 
                $b{$v[0]}{$i}=$v[$tf{$tpm_fpkm_expectedCount}];
                 
            }
            close IN;
        }
        my $outfile="${indir}/"."$gene_iso"."_expression_"."$tpm_fpkm_expectedCount".".tsv";
        open OUT, ">$outfile";
        if ($gene_iso ne "isoforms") {
            print OUT "gene\\ttranscript";
        } else {
            print OUT "transcript\\tgene";
        }
    
        for(my $j=1;$j<=$i;$j++) {
            print OUT "\\t$a[$j]";
        }
        print OUT "\\n";
    
        foreach my $key (keys %b) {
            print OUT "$key\\t$c{$key}";
            for(my $j=1;$j<=$i;$j++){
                print OUT "\\t$b{$key}{$j}";
            }
            print OUT "\\n";
        }
        close OUT;
    }
}

'''
}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_Salmon_Prepare_DESeq2 {

input:
 file counts from g276_15_outputFile00_g302_24
 file groups_file from g_295_1_g302_24
 file compare_file from g_294_2_g302_24
 val run_DESeq2 from g_309_3_g302_24

output:
 file "DE_reports"  into g302_24_outputFile00_g302_37
 val "_des"  into g302_24_postfix10_g302_33
 file "DE_reports/outputs/*_all_deseq2_results.tsv"  into g302_24_outputFile21_g302_33

container 'quay.io/viascientific/de_module:4.0'

when:
run_DESeq2 == 'yes'

script:

feature_type = params.DE_module_STAR_Salmon_Prepare_DESeq2.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_distribution
include_all2all = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_all2all
include_pca = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_pca

filter_type = params.DE_module_STAR_Salmon_Prepare_DESeq2.filter_type
min_count = params.DE_module_STAR_Salmon_Prepare_DESeq2.min_count
min_samples = params.DE_module_STAR_Salmon_Prepare_DESeq2.min_samples
min_counts_per_sample = params.DE_module_STAR_Salmon_Prepare_DESeq2.min_counts_per_sample
excluded_events = params.DE_module_STAR_Salmon_Prepare_DESeq2.excluded_events

include_batch_correction = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_batch_correction
batch_correction_column = params.DE_module_STAR_Salmon_Prepare_DESeq2.batch_correction_column
batch_correction_group_column = params.DE_module_STAR_Salmon_Prepare_DESeq2.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_STAR_Salmon_Prepare_DESeq2.batch_normalization_algorithm

transformation = params.DE_module_STAR_Salmon_Prepare_DESeq2.transformation
pca_color = params.DE_module_STAR_Salmon_Prepare_DESeq2.pca_color
pca_shape = params.DE_module_STAR_Salmon_Prepare_DESeq2.pca_shape
pca_fill = params.DE_module_STAR_Salmon_Prepare_DESeq2.pca_fill
pca_transparency = params.DE_module_STAR_Salmon_Prepare_DESeq2.pca_transparency
pca_label = params.DE_module_STAR_Salmon_Prepare_DESeq2.pca_label

include_deseq2 = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_deseq2
input_mode = params.DE_module_STAR_Salmon_Prepare_DESeq2.input_mode
design = params.DE_module_STAR_Salmon_Prepare_DESeq2.design
fitType = params.DE_module_STAR_Salmon_Prepare_DESeq2.fitType
use_batch_corrected_in_DE = params.DE_module_STAR_Salmon_Prepare_DESeq2.use_batch_corrected_in_DE
apply_shrinkage = params.DE_module_STAR_Salmon_Prepare_DESeq2.apply_shrinkage
shrinkage_type = params.DE_module_STAR_Salmon_Prepare_DESeq2.shrinkage_type
include_volcano = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_volcano
include_ma = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_ma
include_heatmap = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_heatmap

padj_significance_cutoff = params.DE_module_STAR_Salmon_Prepare_DESeq2.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_STAR_Salmon_Prepare_DESeq2.fc_significance_cutoff
padj_floor = params.DE_module_STAR_Salmon_Prepare_DESeq2.padj_floor
fc_ceiling = params.DE_module_STAR_Salmon_Prepare_DESeq2.fc_ceiling

convert_names = params.DE_module_STAR_Salmon_Prepare_DESeq2.convert_names
count_file_names = params.DE_module_STAR_Salmon_Prepare_DESeq2.count_file_names
converted_name = params.DE_module_STAR_Salmon_Prepare_DESeq2.converted_name
org_db = params.DE_module_STAR_Salmon_Prepare_DESeq2.org_db
num_labeled = params.DE_module_STAR_Salmon_Prepare_DESeq2.num_labeled
highlighted_genes = params.DE_module_STAR_Salmon_Prepare_DESeq2.highlighted_genes
include_volcano_highlighted = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_volcano_highlighted
include_ma_highlighted = params.DE_module_STAR_Salmon_Prepare_DESeq2.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name, org_db},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm, use_batch_corrected_in_DE},{include_batch_correction="false"},{include_deseq2="true", design, fitType, apply_shrinkage, shrinkage_type, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, org_db, num_labeled, highlighted_genes},{include_deseq2="false"},{apply_shrinkage="true", shrinkage_type},{apply_shrinkage="false"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples, min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{design, fitType, use_batch_corrected_in_DE, apply_shrinkage, shrinkage_type},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name, org_db},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_deseq2 = include_deseq2 == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
apply_shrinkage = apply_shrinkage == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir reports
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_DESeq2.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-DESeq2 ${include_deseq2} --input-mode ${input_mode} --design '${design}' --fitType ${fitType} --use-batch-correction-in-DE ${use_batch_corrected_in_DE} --apply-shrinkage ${apply_shrinkage} --shrinkage-type ${shrinkage_type} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --org-db ${org_db} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_Salmon_Prepare_LimmaVoom {

input:
 file counts from g276_15_outputFile00_g302_25
 file groups_file from g_295_1_g302_25
 file compare_file from g_294_2_g302_25
 val run_limmaVoom from g_359_3_g302_25

output:
 file "DE_reports"  into g302_25_outputFile00_g302_39
 val "_lv"  into g302_25_postfix10_g302_41
 file "DE_reports/outputs/*_all_limmaVoom_results.tsv"  into g302_25_outputFile21_g302_41

container 'quay.io/viascientific/de_module:4.0'

when:
run_limmaVoom == 'yes'

script:

feature_type = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_distribution
include_all2all = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_all2all
include_pca = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_pca

filter_type = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.filter_type
min_count = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.min_count
min_samples = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.min_samples
min_counts_per_sample = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.min_counts_per_sample
excluded_events = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.excluded_events

include_batch_correction = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_batch_correction
batch_correction_column = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.batch_correction_column
batch_correction_group_column = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.batch_normalization_algorithm

transformation = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.transformation
pca_color = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.pca_color
pca_shape = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.pca_shape
pca_fill = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.pca_fill
pca_transparency = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.pca_transparency
pca_label = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.pca_label

include_limma = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_limma
use_batch_corrected_in_DE = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.use_batch_corrected_in_DE
normalization_method = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.normalization_method
logratioTrim = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.logratioTrim
sumTrim = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.sumTrim
Acutoff = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.Acutoff
doWeighting = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.doWeighting
p = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.p
include_volcano = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_volcano
include_ma = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_ma
include_heatmap = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_heatmap

padj_significance_cutoff = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.fc_significance_cutoff
padj_floor = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.padj_floor
fc_ceiling = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.fc_ceiling

convert_names = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.convert_names
count_file_names = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.count_file_names
converted_name = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.converted_name
num_labeled = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.num_labeled
highlighted_genes = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.highlighted_genes
include_volcano_highlighted = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_volcano_highlighted
include_ma_highlighted = params.DE_module_STAR_Salmon_Prepare_LimmaVoom.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm,use_batch_corrected_in_DE},{include_batch_correction="false"},{include_limma="true", normalization_method, logratioTrim, sumTrim, doWeighting, Acutoff, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, num_labeled, highlighted_genes},{include_limma="false"},{normalization_method="TMM", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="TMMwsp", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="RLE"},{normalization_method="upperquartile", p},{normalization_method="none"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples,min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_limma, use_batch_corrected_in_DE},{normalization_method,logratioTrim,sumTrim,doWeighting,Acutoff,p},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_limma = include_limma == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

doWeighting = doWeighting == 'true' ? 'TRUE' : 'FALSE'
TMM_args = normalization_method.equals('TMM') || normalization_method.equals('TMMwsp') ? '--logratio-trim ' + logratioTrim + ' --sum-trim ' + sumTrim + ' --do-weighting ' + doWeighting + ' --A-cutoff="' + Acutoff + '"' : ''
upperquartile_args = normalization_method.equals('upperquartile') ? '--p ' + p : ''

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_limmaVoom.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-limma ${include_limma} \
--use-batch-correction-in-DE ${use_batch_corrected_in_DE} --normalization-method ${normalization_method} ${TMM_args} ${upperquartile_args} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}

//* params.salmon_index =  ""  //* @input
//* params.genome_sizes =  ""  //* @input
//* params.gtf =  ""  //* @input
//* @style @multicolumn:{fragment_length,standard_deviation} @condition:{single_or_paired_end_reads="single", fragment_length,standard_deviation}, {single_or_paired_end_reads="pair"}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 4
    $MEMORY = 20 
}
//* platform
//* platform
//* autofill


process Salmon_module_salmon_quant {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /salmon_${name}$/) "salmon/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.bam$/) "salmon/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.bam.bai$/) "salmon/$filename"}
input:
 val mate from g_347_mate10_g268_44
 set val(name), file(reads) from g256_46_reads01_g268_44
 file salmon_index from g268_43_salmon_index02_g268_44
 file gtf from g245_54_gtfFile03_g268_44
 file genome_sizes from g245_54_genomeSizes24_g268_44
 file genome from g245_54_genome15_g268_44

output:
 file "salmon_${name}"  into g268_44_outputDir00_g268_45, g268_44_outputDir00_g268_48, g268_44_outputDir016_g_177
 set val(name), file("*.bam") optional true  into g268_44_bam_file11_g274_131, g268_44_bam_file10_g274_121, g268_44_bam_file10_g274_143
 file "*.bam.bai" optional true  into g268_44_bam_bai22

container 'quay.io/viascientific/salmon:1.0'

when:
(params.run_Salmon && (params.run_Salmon == "yes")) || !params.run_Salmon

script:
salmon_parameters = params.Salmon_module_salmon_quant.salmon_parameters
input_reads = mate == "single" ? "-r $reads" : "-1 ${reads[0]} -2 ${reads[1]}"
libType = params.Salmon_module_salmon_quant.libType
//. strandedness = meta.single_end ? 'U' : 'IU'
//     if (meta.strandedness == 'forward') {
//         strandedness = meta.single_end ? 'SF' : 'ISF'
//     } else if (meta.strandedness == 'reverse') {
//         strandedness = meta.single_end ? 'SR' : 'ISR'
"""
# --writeQualities added to support picard CollectMultipleMetrics
salmon quant --geneMap $gtf --threads $task.cpus --libType=$libType --index $salmon_index $input_reads  $salmon_parameters --writeMappings=${name}.sam --writeQualities -o salmon_${name}


if ls ${name}.sam 1> /dev/null 2>&1; then
	samtools view -S -F 4 -b ${name}.sam > ${name}_clean.bam
    samtools sort -o ${name}_sorted.bam ${name}_clean.bam && samtools index ${name}_sorted.bam
    rm ${name}_clean.bam ${name}.sam
fi


if [ -f salmon_${name}/quant.sf ]; then
  mv salmon_${name}/quant.sf  salmon_${name}/abundance_isoforms.tsv
fi


"""

}

//* params.gtf =  ""  //* @input

//* autofill
//* platform
//* platform
//* autofill

process Salmon_module_Salmon_transcript_to_gene_count {

input:
 file outDir from g268_44_outputDir00_g268_48
 file gtf from g245_54_gtfFile01_g268_48

output:
 file newoutDir  into g268_48_outputDir00_g268_47

shell:
newoutDir = "genes_" + outDir
'''
#!/usr/bin/env perl
use strict;
use Getopt::Long;
use IO::File;
use Data::Dumper;

my $gtf_file = "!{gtf}";
my $transcript_matrix_in = "!{outDir}/abundance_isoforms.tsv";
my $transcript_matrix_out = "!{outDir}/abundance_genes.tsv";
open(IN, "<$gtf_file") or die "Can't open $gtf_file.\\n";
my %all_genes; # save gene_id of transcript_id
while(<IN>){
  next if(/^##/); #ignore header
  chomp;
  my %attribs = ();
  my ($chr, $source, $type, $start, $end, $score,
    $strand, $phase, $attributes) = split("\\t");
  my @add_attributes = split(";", $attributes);
  # store ids and additional information in second hash
  foreach my $attr ( @add_attributes ) {
     next unless $attr =~ /^\\s*(.+)\\s(.+)$/;
     my $c_type  = $1;
     my $c_value = $2;
     $c_value =~ s/\\"//g;
     if($c_type  && $c_value){
       if(!exists($attribs{$c_type})){
         $attribs{$c_type} = [];
       }
       push(@{ $attribs{$c_type} }, $c_value);
     }
  }
  #work with the information from the two hashes...
  if(exists($attribs{'transcript_id'}->[0]) && exists($attribs{'gene_id'}->[0])){
    if(!exists($all_genes{$attribs{'transcript_id'}->[0]})){
        $all_genes{$attribs{'transcript_id'}->[0]} = $attribs{'gene_id'}->[0];
    }
  } 
}


# print Dumper \\%all_genes;

#Parse the salmon input file, determine gene IDs for each transcript, and calculate sum TPM values
my %gene_exp;
my %gene_length;
my %samples;
my $ki_fh = IO::File->new($transcript_matrix_in, 'r');
my $header = '';
my $h = 0;
while (my $ki_line = $ki_fh->getline) {
  $h++;
  chomp($ki_line);
  my @ki_entry = split("\\t", $ki_line);
  my $s = 0;
  if ($h == 1){
    $header = $ki_line;
    my $first_col = shift @ki_entry;
    my $second_col = shift @ki_entry;
    foreach my $sample (@ki_entry){
      $s++;
      $samples{$s}{name} = $sample;
    }
    next;
  }
  my $trans_id = shift @ki_entry;
  my $length = shift @ki_entry;
  my $gene_id;
  if ($all_genes{$trans_id}){
    $gene_id = $all_genes{$trans_id};
  }elsif($trans_id =~ /ERCC/){
    $gene_id = $trans_id;
  }else{
    print "\\n\\nCould not identify gene id from trans id: $trans_id\\n\\n";
  }

  $s = 0;
  foreach my $value (@ki_entry){
    $s++;
    $gene_exp{$gene_id}{$s} += $value;
  }
  if ($gene_length{$gene_id}){
    $gene_length{$gene_id} = $length if ($length > $gene_length{$gene_id});
  }else{
    $gene_length{$gene_id} = $length;
  }

}
$ki_fh->close;

my $ko_fh = IO::File->new($transcript_matrix_out, 'w');
unless ($ko_fh) { die('Failed to open file: '. $transcript_matrix_out); }

print $ko_fh "$header\\n";
foreach my $gene_id (sort {$a cmp $b} keys %gene_exp){
  print $ko_fh "$gene_id\\t$gene_length{$gene_id}\\t";
  my @vals;
  foreach my $s (sort {$a <=> $b} keys %samples){
     push(@vals, $gene_exp{$gene_id}{$s});
  }
  my $val_string = join("\\t", @vals);
  print $ko_fh "$val_string\\n";
}


$ko_fh->close;
if (checkFile("!{outDir}")){
	rename ("!{outDir}", "!{newoutDir}");
}

sub checkFile {
    my ($file) = @_;
    print "$file\\n";
    return 1 if ( -e $file );
    return 0;
}

'''
}

//* params.gtf =  ""  //* @input

//* autofill
//* platform
//* platform
//* autofill

process Salmon_module_Salmon_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "salmon_count/$filename"}
input:
 file salmonOut from g268_48_outputDir00_g268_47.collect()
 file gtf from g245_54_gtfFile01_g268_47

output:
 file "*.tsv"  into g268_47_outputFile00_g304_25, g268_47_outputFile00_g304_24

shell:
'''
#!/usr/bin/env perl
use Data::Dumper;
use strict;

### Parse gtf file
my $gtf_file = "!{gtf}";
open(IN, "<$gtf_file") or die "Can't open $gtf_file.\\n";
my %all_genes; # save gene_id of transcript_id
my %all_trans; # map transcript_id of genes
while(<IN>){
  next if(/^##/); #ignore header
  chomp;
  my %attribs = ();
  my ($chr, $source, $type, $start, $end, $score,
    $strand, $phase, $attributes) = split("\\t");
  my @add_attributes = split(";", $attributes);
  # store ids and additional information in second hash
  foreach my $attr ( @add_attributes ) {
     next unless $attr =~ /^\\s*(.+)\\s(.+)$/;
     my $c_type  = $1;
     my $c_value = $2;
     $c_value =~ s/\\"//g;
     if($c_type  && $c_value){
       if(!exists($attribs{$c_type})){
         $attribs{$c_type} = [];
       }
       push(@{ $attribs{$c_type} }, $c_value);
     }
  }
  #work with the information from the two hashes...
  if(exists($attribs{'transcript_id'}->[0]) && exists($attribs{'gene_id'}->[0])){
    if(!exists($all_genes{$attribs{'transcript_id'}->[0]})){
        $all_genes{$attribs{'transcript_id'}->[0]} = $attribs{'gene_id'}->[0];
    }
    if(!exists($all_trans{$attribs{'gene_id'}->[0]})){
        $all_trans{$attribs{'gene_id'}->[0]} = $attribs{'transcript_id'}->[0];
    } else {
    	if (index($all_trans{$attribs{'gene_id'}->[0]}, $attribs{'transcript_id'}->[0]) == -1) {
			$all_trans{$attribs{'gene_id'}->[0]} = $all_trans{$attribs{'gene_id'}->[0]} . "," .$attribs{'transcript_id'}->[0];
		}
    	
    }
  } 
}


print Dumper \\%all_trans;



#### Create summary table

my %tf = (
        expected_count => 4,
        tpm => 3
    );

my $indir = $ENV{'PWD'};
my $outdir = $ENV{'PWD'};

my @gene_iso_ar = ("genes","isoforms");
my @tpm_fpkm_expectedCount_ar = ("expected_count", "tpm");
for(my $l = 0; $l <= $#gene_iso_ar; $l++) {
    my $gene_iso = $gene_iso_ar[$l];
    for(my $ll = 0; $ll <= $#tpm_fpkm_expectedCount_ar; $ll++) {
        my $tpm_fpkm_expectedCount = $tpm_fpkm_expectedCount_ar[$ll];

        opendir D, $indir or die "Could not open $indir\\n";
        my @alndirs = sort { $a cmp $b } grep /^genes_salmon_/, readdir(D);
        closedir D;
    
        my @a=();
        my %b=();
        my %c=();
        my $i=0;
        foreach my $d (@alndirs){ 
            my $dir = "${indir}/$d";
            print $d."\\n";
            my $libname=$d;
            $libname=~s/genes_salmon_//;
            $i++;
            $a[$i]=$libname;
            open IN,"${dir}/abundance_${gene_iso}.tsv";
            $_=<IN>;
            while(<IN>)
            {
                my @v=split; 
                # $v[0] -> transcript_id
                # $all_genes{$v[0]} -> $gene_id
                if ($gene_iso eq "isoforms"){
                	$c{$v[0]}=$all_genes{$v[0]};
                } elsif ($gene_iso eq "genes"){
                	$c{$v[0]}=$all_trans{$v[0]};
                } 
                $b{$v[0]}{$i}=$v[$tf{$tpm_fpkm_expectedCount}];
                 
            }
            close IN;
        }
        my $outfile="${indir}/"."$gene_iso"."_expression_"."$tpm_fpkm_expectedCount".".tsv";
        open OUT, ">$outfile";
        if ($gene_iso ne "isoforms") {
            print OUT "gene\\ttranscript";
        } else {
            print OUT "transcript\\tgene";
        }
    
        for(my $j=1;$j<=$i;$j++) {
            print OUT "\\t$a[$j]";
        }
        print OUT "\\n";
    
        foreach my $key (keys %b) {
            print OUT "$key\\t$c{$key}";
            for(my $j=1;$j<=$i;$j++){
                print OUT "\\t$b{$key}{$j}";
            }
            print OUT "\\n";
        }
        close OUT;
    }
}

'''
}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_Salmon_Prepare_DESeq2 {

input:
 file counts from g268_47_outputFile00_g304_24
 file groups_file from g_295_1_g304_24
 file compare_file from g_294_2_g304_24
 val run_DESeq2 from g_311_3_g304_24

output:
 file "DE_reports"  into g304_24_outputFile00_g304_37
 val "_des"  into g304_24_postfix10_g304_33
 file "DE_reports/outputs/*_all_deseq2_results.tsv"  into g304_24_outputFile21_g304_33

container 'quay.io/viascientific/de_module:4.0'

when:
run_DESeq2 == 'yes'

script:

feature_type = params.DE_module_Salmon_Prepare_DESeq2.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_Salmon_Prepare_DESeq2.include_distribution
include_all2all = params.DE_module_Salmon_Prepare_DESeq2.include_all2all
include_pca = params.DE_module_Salmon_Prepare_DESeq2.include_pca

filter_type = params.DE_module_Salmon_Prepare_DESeq2.filter_type
min_count = params.DE_module_Salmon_Prepare_DESeq2.min_count
min_samples = params.DE_module_Salmon_Prepare_DESeq2.min_samples
min_counts_per_sample = params.DE_module_Salmon_Prepare_DESeq2.min_counts_per_sample
excluded_events = params.DE_module_Salmon_Prepare_DESeq2.excluded_events

include_batch_correction = params.DE_module_Salmon_Prepare_DESeq2.include_batch_correction
batch_correction_column = params.DE_module_Salmon_Prepare_DESeq2.batch_correction_column
batch_correction_group_column = params.DE_module_Salmon_Prepare_DESeq2.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_Salmon_Prepare_DESeq2.batch_normalization_algorithm

transformation = params.DE_module_Salmon_Prepare_DESeq2.transformation
pca_color = params.DE_module_Salmon_Prepare_DESeq2.pca_color
pca_shape = params.DE_module_Salmon_Prepare_DESeq2.pca_shape
pca_fill = params.DE_module_Salmon_Prepare_DESeq2.pca_fill
pca_transparency = params.DE_module_Salmon_Prepare_DESeq2.pca_transparency
pca_label = params.DE_module_Salmon_Prepare_DESeq2.pca_label

include_deseq2 = params.DE_module_Salmon_Prepare_DESeq2.include_deseq2
input_mode = params.DE_module_Salmon_Prepare_DESeq2.input_mode
design = params.DE_module_Salmon_Prepare_DESeq2.design
fitType = params.DE_module_Salmon_Prepare_DESeq2.fitType
use_batch_corrected_in_DE = params.DE_module_Salmon_Prepare_DESeq2.use_batch_corrected_in_DE
apply_shrinkage = params.DE_module_Salmon_Prepare_DESeq2.apply_shrinkage
shrinkage_type = params.DE_module_Salmon_Prepare_DESeq2.shrinkage_type
include_volcano = params.DE_module_Salmon_Prepare_DESeq2.include_volcano
include_ma = params.DE_module_Salmon_Prepare_DESeq2.include_ma
include_heatmap = params.DE_module_Salmon_Prepare_DESeq2.include_heatmap

padj_significance_cutoff = params.DE_module_Salmon_Prepare_DESeq2.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_Salmon_Prepare_DESeq2.fc_significance_cutoff
padj_floor = params.DE_module_Salmon_Prepare_DESeq2.padj_floor
fc_ceiling = params.DE_module_Salmon_Prepare_DESeq2.fc_ceiling

convert_names = params.DE_module_Salmon_Prepare_DESeq2.convert_names
count_file_names = params.DE_module_Salmon_Prepare_DESeq2.count_file_names
converted_name = params.DE_module_Salmon_Prepare_DESeq2.converted_name
org_db = params.DE_module_Salmon_Prepare_DESeq2.org_db
num_labeled = params.DE_module_Salmon_Prepare_DESeq2.num_labeled
highlighted_genes = params.DE_module_Salmon_Prepare_DESeq2.highlighted_genes
include_volcano_highlighted = params.DE_module_Salmon_Prepare_DESeq2.include_volcano_highlighted
include_ma_highlighted = params.DE_module_Salmon_Prepare_DESeq2.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name, org_db},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm, use_batch_corrected_in_DE},{include_batch_correction="false"},{include_deseq2="true", design, fitType, apply_shrinkage, shrinkage_type, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, org_db, num_labeled, highlighted_genes},{include_deseq2="false"},{apply_shrinkage="true", shrinkage_type},{apply_shrinkage="false"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples, min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{design, fitType, use_batch_corrected_in_DE, apply_shrinkage, shrinkage_type},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name, org_db},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_deseq2 = include_deseq2 == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
apply_shrinkage = apply_shrinkage == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir reports
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_DESeq2.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-DESeq2 ${include_deseq2} --input-mode ${input_mode} --design '${design}' --fitType ${fitType} --use-batch-correction-in-DE ${use_batch_corrected_in_DE} --apply-shrinkage ${apply_shrinkage} --shrinkage-type ${shrinkage_type} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --org-db ${org_db} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_Salmon_Prepare_LimmaVoom {

input:
 file counts from g268_47_outputFile00_g304_25
 file groups_file from g_295_1_g304_25
 file compare_file from g_294_2_g304_25
 val run_limmaVoom from g_361_3_g304_25

output:
 file "DE_reports"  into g304_25_outputFile00_g304_39
 val "_lv"  into g304_25_postfix10_g304_41
 file "DE_reports/outputs/*_all_limmaVoom_results.tsv"  into g304_25_outputFile21_g304_41

container 'quay.io/viascientific/de_module:4.0'

when:
run_limmaVoom == 'yes'

script:

feature_type = params.DE_module_Salmon_Prepare_LimmaVoom.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_Salmon_Prepare_LimmaVoom.include_distribution
include_all2all = params.DE_module_Salmon_Prepare_LimmaVoom.include_all2all
include_pca = params.DE_module_Salmon_Prepare_LimmaVoom.include_pca

filter_type = params.DE_module_Salmon_Prepare_LimmaVoom.filter_type
min_count = params.DE_module_Salmon_Prepare_LimmaVoom.min_count
min_samples = params.DE_module_Salmon_Prepare_LimmaVoom.min_samples
min_counts_per_sample = params.DE_module_Salmon_Prepare_LimmaVoom.min_counts_per_sample
excluded_events = params.DE_module_Salmon_Prepare_LimmaVoom.excluded_events

include_batch_correction = params.DE_module_Salmon_Prepare_LimmaVoom.include_batch_correction
batch_correction_column = params.DE_module_Salmon_Prepare_LimmaVoom.batch_correction_column
batch_correction_group_column = params.DE_module_Salmon_Prepare_LimmaVoom.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_Salmon_Prepare_LimmaVoom.batch_normalization_algorithm

transformation = params.DE_module_Salmon_Prepare_LimmaVoom.transformation
pca_color = params.DE_module_Salmon_Prepare_LimmaVoom.pca_color
pca_shape = params.DE_module_Salmon_Prepare_LimmaVoom.pca_shape
pca_fill = params.DE_module_Salmon_Prepare_LimmaVoom.pca_fill
pca_transparency = params.DE_module_Salmon_Prepare_LimmaVoom.pca_transparency
pca_label = params.DE_module_Salmon_Prepare_LimmaVoom.pca_label

include_limma = params.DE_module_Salmon_Prepare_LimmaVoom.include_limma
use_batch_corrected_in_DE = params.DE_module_Salmon_Prepare_LimmaVoom.use_batch_corrected_in_DE
normalization_method = params.DE_module_Salmon_Prepare_LimmaVoom.normalization_method
logratioTrim = params.DE_module_Salmon_Prepare_LimmaVoom.logratioTrim
sumTrim = params.DE_module_Salmon_Prepare_LimmaVoom.sumTrim
Acutoff = params.DE_module_Salmon_Prepare_LimmaVoom.Acutoff
doWeighting = params.DE_module_Salmon_Prepare_LimmaVoom.doWeighting
p = params.DE_module_Salmon_Prepare_LimmaVoom.p
include_volcano = params.DE_module_Salmon_Prepare_LimmaVoom.include_volcano
include_ma = params.DE_module_Salmon_Prepare_LimmaVoom.include_ma
include_heatmap = params.DE_module_Salmon_Prepare_LimmaVoom.include_heatmap

padj_significance_cutoff = params.DE_module_Salmon_Prepare_LimmaVoom.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_Salmon_Prepare_LimmaVoom.fc_significance_cutoff
padj_floor = params.DE_module_Salmon_Prepare_LimmaVoom.padj_floor
fc_ceiling = params.DE_module_Salmon_Prepare_LimmaVoom.fc_ceiling

convert_names = params.DE_module_Salmon_Prepare_LimmaVoom.convert_names
count_file_names = params.DE_module_Salmon_Prepare_LimmaVoom.count_file_names
converted_name = params.DE_module_Salmon_Prepare_LimmaVoom.converted_name
num_labeled = params.DE_module_Salmon_Prepare_LimmaVoom.num_labeled
highlighted_genes = params.DE_module_Salmon_Prepare_LimmaVoom.highlighted_genes
include_volcano_highlighted = params.DE_module_Salmon_Prepare_LimmaVoom.include_volcano_highlighted
include_ma_highlighted = params.DE_module_Salmon_Prepare_LimmaVoom.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm,use_batch_corrected_in_DE},{include_batch_correction="false"},{include_limma="true", normalization_method, logratioTrim, sumTrim, doWeighting, Acutoff, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, num_labeled, highlighted_genes},{include_limma="false"},{normalization_method="TMM", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="TMMwsp", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="RLE"},{normalization_method="upperquartile", p},{normalization_method="none"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples,min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_limma, use_batch_corrected_in_DE},{normalization_method,logratioTrim,sumTrim,doWeighting,Acutoff,p},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_limma = include_limma == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

doWeighting = doWeighting == 'true' ? 'TRUE' : 'FALSE'
TMM_args = normalization_method.equals('TMM') || normalization_method.equals('TMMwsp') ? '--logratio-trim ' + logratioTrim + ' --sum-trim ' + sumTrim + ' --do-weighting ' + doWeighting + ' --A-cutoff="' + Acutoff + '"' : ''
upperquartile_args = normalization_method.equals('upperquartile') ? '--p ' + p : ''

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_limmaVoom.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-limma ${include_limma} \
--use-batch-correction-in-DE ${use_batch_corrected_in_DE} --normalization-method ${normalization_method} ${TMM_args} ${upperquartile_args} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}

//* autofill
//* platform
//* platform
//* autofill

process Salmon_module_Salmon_Alignment_Summary {

input:
 file kallistoDir from g268_44_outputDir00_g268_45.collect()

output:
 file "salmon_alignment_sum.tsv"  into g268_45_outFileTSV011_g_198

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;
my $indir = $ENV{'PWD'};

opendir D, $indir or die "Could not open $indir";
my @alndirs = sort { $a cmp $b } grep /^salmon_/, readdir(D);
closedir D;

my @a=();
my %b=();
my %c=();
my $i=0;
my @headers = ();
my %tsv;
foreach my $d (@alndirs){
    my $dir = "${indir}/$d";
    my $libname=$d;
    $libname=~s/salmon_//;
    my $multimapped;
    my $aligned;
    my $total;
    my $unique;
    
    # eg. [quant] processed 24,788 reads, 19,238 reads pseudoaligned
	chomp($total   = `cat ${dir}/aux_info/meta_info.json | grep 'num_processed' | sed 's/,//g' | sed 's/"//g' | sed 's/://g' | awk '{sum+=\\$2} END {print sum}'`);
	chomp($aligned = `cat ${dir}/aux_info/meta_info.json | grep 'num_mapped' | sed 's/,//g' | sed 's/"//g' | sed 's/://g' | awk '{sum+=\\$2} END {print sum}'`);
	chomp($unique = `cat ${dir}/aux_info/ambig_info.tsv  | awk -F"\\t" '{ sum+=\\$1} END {print sum}'`);
    $tsv{$libname}=[$libname, $total];
    push(@{$tsv{$libname}}, $aligned);
    push(@{$tsv{$libname}}, $unique);
}

push(@headers, "Sample");
push(@headers, "Total Reads");
push(@headers, "Pseudoaligned Reads (Salmon)");
push(@headers, "Uniquely Mapped Reads (Salmon)");

my @keys = keys %tsv;
my $summary = "salmon_alignment_sum.tsv";
my $header_string = join("\\t", @headers);
`echo "$header_string" > $summary`;
foreach my $key (@keys){
    my $values = join("\\t", @{ $tsv{$key} });
        `echo "$values" >> $summary`;
}
'''
}


process BAM_Analysis_Module_Salmon_bam_sort_index {

input:
 set val(name), file(bam) from g268_44_bam_file10_g274_143

output:
 set val(name), file("bam/*.bam"), file("bam/*.bam.bai")  into g274_143_bam_bai00_g274_134, g274_143_bam_bai00_g274_142

when:
params.run_BigWig_Conversion == "yes" || params.run_RSeQC == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}

"""
$runSamtools
mkdir -p bam
mv ${name}_sorted.bam ${name}_sorted.bam.bai bam/.
"""
}



//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 30
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_Salmon_UCSC_BAM2BigWig_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.bw$/) "bigwig_salmon/$filename"}
input:
 set val(name), file(bam), file(bai) from g274_143_bam_bai00_g274_142
 file genomeSizes from g245_54_genomeSizes21_g274_142

output:
 file "*.bw" optional true  into g274_142_outputFileBw00
 file "publish/*.bw" optional true  into g274_142_publishBw10_g274_145

container 'quay.io/biocontainers/deeptools:3.5.4--pyhdfd78af_1'

when:
params.run_BigWig_Conversion == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = ""
    nameFinal = nameAll
} else {
    runSamtools = "mv $bam ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
deeptools_parameters = params.BAM_Analysis_Module_Salmon_UCSC_BAM2BigWig_converter.deeptools_parameters
visualize_bigwig_in_reports = params.BAM_Analysis_Module_Salmon_UCSC_BAM2BigWig_converter.visualize_bigwig_in_reports

"""
$runSamtools
bamCoverage ${deeptools_parameters}  -b ${nameFinal} -o ${name}.bw 

if [ "${visualize_bigwig_in_reports}" == "yes" ]; then
	mkdir -p publish
	mv ${name}.bw publish/.
fi
"""

}



process BAM_Analysis_Module_Salmon_Genome_Browser {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "genome_browser_salmon/$filename"}
input:
 file bigwigs from g274_142_publishBw10_g274_145.collect()
 file group_file from g_295_1_g274_145

output:
 file "*"  into g274_145_bigWig_file00

container 'quay.io/viascientific/python-basics:2.0'


script:
try {
    myVariable = group_file
} catch (MissingPropertyException e) {
    group_file = ""
}

genome_build_short= ""
if (params.genome_build == "mousetest_mm10"){
    genome_build_short= "mm10"
} else if (params.genome_build == "human_hg19_refseq"){
    genome_build_short = "hg19"
} else if (params.genome_build == "human_hg38_gencode_v28"){
    genome_build_short = "hg38"
} else if (params.genome_build == "human_hg38_gencode_v34"){
    genome_build_short = "hg38"
} else if (params.genome_build == "mouse_mm10_refseq"){
    genome_build_short = "mm10"
} else if (params.genome_build == "mouse_mm10_gencode_m25"){
    genome_build_short = "mm10"
} else if (params.genome_build == "rat_rn6_refseq"){
    genome_build_short = "rn6"
} else if (params.genome_build == "rat_rn6_ensembl_v86"){
    genome_build_short = "rn6"
} else if (params.genome_build == "zebrafish_GRCz11_ensembl_v95"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_refseq"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_v4.3.2"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "c_elegans_ce11_ensembl_ws245"){
    genome_build_short = "ce11"
} else if (params.genome_build == "d_melanogaster_dm6_refseq"){
    genome_build_short = "dm6"
} else if (params.genome_build == "s_cerevisiae_sacCer3_refseq"){
    genome_build_short = "sacCer3"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v31"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v51"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "e_coli_ASM584v2_refseq"){
    genome_build_short = "ASM584v2"
} else if (params.genome_build == "dog_canFam3_refseq"){
    genome_build_short = "canFam3"
} 

"""

#!/usr/bin/env python

import glob,json,os,sys,csv,random,shutil,requests,subprocess

def check_url_existence(url):
    try:
        response = requests.head(url, allow_redirects=True)
        # Check if the status code is in the success range (200-399)
        return 200 <= response.status_code < 400
    except requests.ConnectionError:
        return False

def get_lib_val(str, lib):
    for key, value in lib.items():
        if key.lower() in str.lower():
            return value
    return None

def find_and_move_folders_with_bw_files(start_dir):
    bigwig_dir = "bigwigs"
    if os.path.exists(bigwig_dir)!=True:
        os.makedirs(bigwig_dir)
    subprocess.getoutput("cp ${bigwigs} bigwigs/. ")

def Generate_HubFile(groupfile):
    # hub.txt
    Hub = open("hub.txt", "w")
    Hub.write("hub UCSCHub \\n")
    Hub.write("shortLabel UCSCHub \\n")
    Hub.write("longLabel UCSCHub \\n")
    Hub.write("genomesFile genomes.txt \\n")
    Hub.write("email support@viascientific.com \\n")
    Hub.write("\\n")
    Hub.close()

    # genomes.txt
    genomes = open("genomes.txt", "w")
    genomes.write("genome ${genome_build_short} \\n")
    genomes.write("trackDb bigwigs/trackDb.txt \\n")
    genomes.write("\\n")
    genomes.close()
    #trackDb = open("bigwigs/trackDb.txt", "w")
    path = r'bigwigs/*.bw'
    files = glob.glob(path)
    files.sort()
    sample={}
    for i in files:
        temp=i.split('.')[0]
        temp=temp.replace('bigwigs/','')
        if temp in sample.keys():
            sample[temp].append(i.replace('bigwigs/',''))
        else:
            sample[temp]=[]
            sample[temp].append(i.replace('bigwigs/',''))
    second_layer_indent=" "*2
    third_layer_indent = " " * 14
    if groupfile == "" or os.stat(groupfile).st_size == 0:
        #No GroupFile
        trackDb = open("bigwigs/trackDb.txt", "w")
        for i in sample.keys():
            trackDb.write('track %s\\n' %i)
            trackDb.write('shortLabel %s\\n' %i)
            trackDb.write('longLabel %s\\n' %i)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for j in sample[i]:
                trackDb.write(second_layer_indent+'track %s\\n' % j)
                trackDb.write(second_layer_indent+'bigDataUrl %s\\n' % j)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %i)
                trackDb.write('\\n')
        trackDb.close()
    else:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_to_sample={}
        for cond in condition:
            if cond not in condition_to_sample.keys():
                condition_to_sample[cond]=[]
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
            else:
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
        trackDb = open("bigwigs/trackDb.txt", "w")
        for cond in condition:
            trackDb.write('track %s\\n' %cond)
            trackDb.write('shortLabel %s\\n' %cond)
            trackDb.write('longLabel %s\\n' %cond)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for samp in condition_to_sample[cond]:
                trackDb.write(second_layer_indent+'track %s\\n' % samp)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'view Signal \\n')
                trackDb.write(second_layer_indent+'visibility full \\n')
                trackDb.write(second_layer_indent+'viewLimits 0:20 \\n')
                trackDb.write(second_layer_indent+'autoScale on \\n')
                trackDb.write(second_layer_indent+'maxHeightPixels 128:20:8 \\n')
                trackDb.write(second_layer_indent+'configurable on \\n')
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %cond)
                trackDb.write('\\n')
                for file in sample[samp]:
                    trackDb.write(third_layer_indent + 'track %s\\n' % file)
                    trackDb.write(third_layer_indent + 'bigDataUrl %s\\n' % file)
                    trackDb.write(third_layer_indent + 'shortLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'longLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'type bigWig\\n')
                    trackDb.write(third_layer_indent+'parent %s\\n' %samp)

                    trackDb.write('\\n')

        trackDb.close()


def Generating_Json_files(groupfile):
    publishWebDir = '{{DNEXT_WEB_REPORT_DIR}}/' + 'genome_browser_salmon' + "/bigwigs"
    locusLib = {}
    # MYC locations
    locusLib["hg19"] = "chr8:128,746,315-128,755,680"
    locusLib["hg38"] = "chr8:127,733,434-127,744,951"
    locusLib["mm10"] = "chr15:61,983,341-61,992,361"
    locusLib["rn6"] = "chr7:102,584,313-102,593,240"
    locusLib["dm6"] = "chrX:3,371,159-3,393,697"
    locusLib["canFam3"] = "chr13:25,198,772-25,207,309"

    cytobandLib = {}
    cytobandLib["hg19"] = "https://igv-genepattern-org.s3.amazonaws.com/genomes/seq/hg19/cytoBand.txt"
    cytobandLib["hg38"] = "https://s3.amazonaws.com/igv.org.genomes/hg38/annotations/cytoBandIdeo.txt.gz"
    cytobandLib["mm10"] = "https://s3.amazonaws.com/igv.broadinstitute.org/annotations/mm10/cytoBandIdeo.txt.gz"
    cytobandLib["rn6"] = "https://s3.amazonaws.com/igv.org.genomes/rn6/cytoBand.txt.gz"
    cytobandLib["dm6"] = "https://s3.amazonaws.com/igv.org.genomes/dm6/cytoBandIdeo.txt.gz"
    cytobandLib["ce11"] = "https://s3.amazonaws.com/igv.org.genomes/ce11/cytoBandIdeo.txt.gz"
    cytobandLib["canFam3"] = "https://s3.amazonaws.com/igv.org.genomes/canFam3/cytoBandIdeo.txt.gz"

    # Get the basename of the original path
    gtf_source_base_name = os.path.basename("${params.gtf_source}")
    gtf_source_sorted = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz")
    print(gtf_source_sorted)
    gtf_source_sorted_index = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz.tbi")
    print(gtf_source_sorted_index)


    data = {}
    data["reference"] = {}
    data["reference"]["id"] = "${params.genome_build}"
    data["reference"]["name"] = "${params.genome_build}"
    data["reference"]["fastaURL"] = "${params.genome_source}"
    data["reference"]["indexURL"] = "${params.genome_source}.fai"
    cytobandurl = get_lib_val("${params.genome_build}", cytobandLib)
    locusStr = get_lib_val("${params.genome_build}", locusLib)
    if cytobandurl is not None:
        data["reference"]["cytobandURL"] = cytobandurl
    if locusStr is not None:
        data["locus"] = []
        data["locus"].append(locusStr)
    data["tracks"] = []
    # prepare gtf Track
    gtfTrack = {}
    gtfTrack["name"] = "${params.genome_build}"
    gtfTrack["gtf"] = "gtf"
    if check_url_existence(gtf_source_sorted):
        gtfTrack["url"] = gtf_source_sorted
    else:
        gtfTrack["url"] = "${params.gtf_source}"
    if check_url_existence(gtf_source_sorted_index):
        gtfTrack["indexURL"] = gtf_source_sorted_index

    # prepare cytobands Track
    if groupfile and os.path.isfile(groupfile) and os.stat(groupfile).st_size != 0:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_color_dict = dict()
        for cond in condition:
            r = lambda: random.randint(0, 255)
            condition_color_dict[cond] = '#%02X%02X%02X' % (r(), r(), r())

    # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                if basename in data_parsed:
                    newdata[file] = data_parsed[basename]
                    newdata[file]["fullname"] = file

        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": condition_color_dict[newdata[j]['group']],
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()
    else:

        # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                newdata[file] = basename
                newdata[file] = file
        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            r = lambda: random.randint(0, 255)
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": '#%02X%02X%02X' % (r(), r(), r()),
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()

if __name__ == "__main__":
    find_and_move_folders_with_bw_files(".")
    Generate_HubFile(groupfile="${group_file}")
    Generating_Json_files(groupfile="${group_file}")

"""
}

//* params.bed =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 10
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_Salmon_RSeQC {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "rseqc_salmon/$filename"}
input:
 set val(name), file(bam), file(bai) from g274_143_bam_bai00_g274_134
 file bed from g245_54_bed31_g274_134
 val mate from g_347_mate12_g274_134

output:
 file "*"  into g274_134_outputFileOut00

container 'quay.io/viascientific/rseqc:1.0'

when:
(params.run_RSeQC && (params.run_RSeQC == "yes")) || !params.run_RSeQC

script:
run_bam_stat = params.BAM_Analysis_Module_Salmon_RSeQC.run_bam_stat
run_read_distribution = params.BAM_Analysis_Module_Salmon_RSeQC.run_read_distribution
run_inner_distance = params.BAM_Analysis_Module_Salmon_RSeQC.run_inner_distance
run_junction_annotation = params.BAM_Analysis_Module_Salmon_RSeQC.run_junction_annotation
run_junction_saturation = params.BAM_Analysis_Module_Salmon_RSeQC.run_junction_saturation
//run_geneBody_coverage and run_infer_experiment needs subsampling
run_geneBody_coverage = params.BAM_Analysis_Module_Salmon_RSeQC.run_geneBody_coverage
run_infer_experiment = params.BAM_Analysis_Module_Salmon_RSeQC.run_infer_experiment
"""
if [ "$run_bam_stat" == "true" ]; then bam_stat.py  -i ${bam} > ${name}.bam_stat.txt; fi
if [ "$run_read_distribution" == "true" ]; then read_distribution.py  -i ${bam} -r ${bed}> ${name}.read_distribution.out; fi


if [ "$run_infer_experiment" == "true" -o "$run_geneBody_coverage" == "true" ]; then
	numAlignedReads=\$(samtools view -c -F 4 $bam)

	if [ "\$numAlignedReads" -gt 1000000 ]; then
    	echo "Read number is greater than 1000000. Subsampling..."
    	finalRead=1000000
    	fraction=\$(samtools idxstats  $bam | cut -f3 | awk -v ct=\$finalRead 'BEGIN {total=0} {total += \$1} END {print ct/total}')
    	samtools view -b -s \${fraction} $bam > ${name}_sampled.bam
    	samtools index ${name}_sampled.bam
    	if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i ${name}_sampled.bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i ${name}_sampled.bam  -r $bed -o ${name}; fi
	else
		if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i $bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i $bam  -r $bed -o ${name}; fi
	fi

fi


if [ "${mate}" == "pair" ]; then
	if [ "$run_inner_distance" == "true" ]; then inner_distance.py -i $bam  -r $bed -o ${name}.inner_distance > stdout.txt; fi
	if [ "$run_inner_distance" == "true" ]; then head -n 2 stdout.txt > ${name}.inner_distance_mean.txt; fi
fi
if [ "$run_junction_annotation" == "true" ]; then junction_annotation.py -i $bam  -r $bed -o ${name}.junction_annotation 2> ${name}.junction_annotation.log; fi
if [ "$run_junction_saturation" == "true" ]; then junction_saturation.py -i $bam  -r $bed -o ${name}; fi
if [ -e class.log ] ; then mv class.log ${name}_class.log; fi
if [ -e log.txt ] ; then mv log.txt ${name}_log.txt; fi
if [ -e stdout.txt ] ; then mv stdout.txt ${name}_stdout.txt; fi


"""

}

//* params.pdfbox_path =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 32
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 240
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "short"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 500
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
}
//* platform
//* autofill

process BAM_Analysis_Module_Salmon_Picard {

input:
 set val(name), file(bam) from g268_44_bam_file10_g274_121

output:
 file "*_metrics"  into g274_121_outputFileOut00_g274_82
 file "results/*.pdf"  into g274_121_outputFilePdf12_g274_82

container 'quay.io/viascientific/picard:1.0'

when:
(params.run_Picard_CollectMultipleMetrics && (params.run_Picard_CollectMultipleMetrics == "yes")) || !params.run_Picard_CollectMultipleMetrics

script:
"""
picard CollectMultipleMetrics OUTPUT=${name}_multiple.out VALIDATION_STRINGENCY=LENIENT INPUT=${bam}
mkdir results && java -jar ${params.pdfbox_path} PDFMerger *.pdf results/${name}_multi_metrics.pdf
"""
}


process BAM_Analysis_Module_Salmon_Picard_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "picard_summary_salmon/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "rseqc_salmon/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /results\/.*.pdf$/) "picard_summary_pdf_salmon/$filename"}
input:
 file picardOut from g274_121_outputFileOut00_g274_82.collect()
 val mate from g_347_mate11_g274_82
 file picardPdf from g274_121_outputFilePdf12_g274_82.collect()

output:
 file "*.tsv"  into g274_82_outputFileTSV00
 file "results/*.pdf"  into g274_82_outputFilePdf11

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage; 
use Data::Dumper;

runCommand("mkdir results && mv *.pdf results/. ");

my $indir = $ENV{'PWD'};
my $outd = $ENV{'PWD'};
my @files = ();
my @outtypes = ("CollectRnaSeqMetrics", "alignment_summary_metrics", "base_distribution_by_cycle_metrics", "insert_size_metrics", "quality_by_cycle_metrics", "quality_distribution_metrics" );

foreach my $outtype (@outtypes)
{
my $ext="_multiple.out";
$ext.=".$outtype" if ($outtype ne "CollectRnaSeqMetrics");
@files = <$indir/*$ext>;

my @rowheaders=();
my @libs=();
my %metricvals=();
my %histvals=();

my $pdffile="";
my $libname="";
foreach my $d (@files){
  my $libname=basename($d, $ext);
  print $libname."\\n";
  push(@libs, $libname); 
  getMetricVals($d, $libname, \\%metricvals, \\%histvals, \\@rowheaders);
}

my $sizemetrics = keys %metricvals;
write_results("$outd/$outtype.stats.tsv", \\@libs,\\%metricvals, \\@rowheaders, "metric") if ($sizemetrics>0);
my $sizehist = keys %histvals;
write_results("$outd/$outtype.hist.tsv", \\@libs,\\%histvals, "none", "nt") if ($sizehist>0);

}

sub write_results
{
  my ($outfile, $libs, $vals, $rowheaders, $name )=@_;
  open(OUT, ">$outfile");
  print OUT "$name\\t".join("\\t", @{$libs})."\\n";
  my $size=0;
  $size=scalar(@{${$vals}{${$libs}[0]}}) if(exists ${$libs}[0] and exists ${$vals}{${$libs}[0]} );
  
  for (my $i=0; $i<$size;$i++)
  { 
    my $rowname=$i;
    $rowname = ${$rowheaders}[$i] if ($name=~/metric/);
    print OUT $rowname;
    foreach my $lib (@{$libs})
    {
      print OUT "\\t".${${$vals}{$lib}}[$i];
    } 
    print OUT "\\n";
  }
  close(OUT);
}

sub getMetricVals{
  my ($filename, $libname, $metricvals, $histvals,$rowheaders)=@_;
  if (-e $filename){
     my $nextisheader=0;
     my $nextisvals=0;
     my $nexthist=0;
     open(IN, $filename);
     while(my $line=<IN>)
     {
       chomp($line);
       @{$rowheaders}=split(/\\t/, $line) if ($nextisheader && !scalar(@{$rowheaders})); 
       if ($nextisvals) {
         @{${$metricvals}{$libname}}=split(/\\t/, $line);
         $nextisvals=0;
       }
       if($nexthist){
          my @vals=split(/[\\s\\t]+/,$line); 
          push(@{${$histvals}{$libname}}, $vals[1]) if (exists $vals[1]);
       }
       $nextisvals=1 if ($nextisheader); $nextisheader=0;
       $nextisheader=1 if ($line=~/METRICS CLASS/);
       $nexthist=1 if ($line=~/normalized_position/);
     } 
  }
  
}


sub runCommand {
	my ($com) = @_;
	if ($com eq ""){
		return "";
    }
    my $error = system(@_);
	if   ($error) { die "Command failed: $error $com\\n"; }
    else          { print "Command successful: $com\\n"; }
}
'''

}

igv_extention_factor = params.BAM_Analysis_Module_Salmon_IGV_BAM2TDF_converter.igv_extention_factor
igv_window_size = params.BAM_Analysis_Module_Salmon_IGV_BAM2TDF_converter.igv_window_size

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 24
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} 
//* platform
//* autofill

process BAM_Analysis_Module_Salmon_IGV_BAM2TDF_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tdf$/) "igvtools_salmon/$filename"}
input:
 val mate from g_347_mate10_g274_131
 set val(name), file(bam) from g268_44_bam_file11_g274_131
 file genomeSizes from g245_54_genomeSizes22_g274_131

output:
 file "*.tdf"  into g274_131_outputFileOut00

when:
(params.run_IGV_TDF_Conversion && (params.run_IGV_TDF_Conversion == "yes")) || !params.run_IGV_TDF_Conversion

script:
pairedText = (params.nucleicAcidType == "dna" && mate == "pair") ? " --pairs " : ""
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
"""
$runSamtools
igvtools count -w ${igv_window_size} -e ${igv_extention_factor} ${pairedText} ${nameFinal} ${name}.tdf ${genomeSizes}
"""
}

//* params.kallisto_index =  ""  //* @input
//* params.genome_sizes =  ""  //* @input
//* params.gtf =  ""  //* @input
//* @style @multicolumn:{fragment_length,standard_deviation} @condition:{single_or_paired_end_reads="single", fragment_length,standard_deviation}, {single_or_paired_end_reads="pair"}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 4
    $MEMORY = 20 
}
//* platform
//* platform
//* autofill


process Kallisto_module_kallisto_quant {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /kallisto_${name}$/) "kallisto/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.bam$/) "kallisto/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.bam.bai$/) "kallisto/$filename"}
input:
 val mate from g_347_mate10_g248_36
 set val(name), file(reads) from g256_46_reads01_g248_36
 file kallisto_index from g248_31_kallisto_index02_g248_36
 file gtf from g245_54_gtfFile03_g248_36
 file genome_sizes from g245_54_genomeSizes24_g248_36
 file genome from g245_54_genome15_g248_36

output:
 file "kallisto_${name}"  into g248_36_outputDir00_g248_22, g248_36_outputDir00_g248_38, g248_36_outputDir013_g_177
 set val(name), file("*.bam") optional true  into g248_36_bam_file11_g255_131, g248_36_bam_file10_g255_121, g248_36_bam_file10_g255_143
 file "*.bam.bai" optional true  into g248_36_bam_bai22

when:
(params.run_Kallisto && (params.run_Kallisto == "yes")) || !params.run_Kallisto

script:

single_or_paired_end_reads = params.Kallisto_module_kallisto_quant.single_or_paired_end_reads
fragment_length = params.Kallisto_module_kallisto_quant.fragment_length
standard_deviation = params.Kallisto_module_kallisto_quant.standard_deviation

kallisto_parameters = params.Kallisto_module_kallisto_quant.kallisto_parameters
genomebam = params.Kallisto_module_kallisto_quant.genomebam

genomebamText = (genomebam.toString() != "false") ? "--genomebam --gtf _${gtf} --chromosomes ${genome_sizes}" : ""
fragment_lengthText = (fragment_length.toString() != "" && mate == "single") ? "-l ${fragment_length}" : ""
standard_deviationText = (standard_deviation.toString() != "" && mate == "single") ? "-s ${standard_deviation}" : ""
"""
filter_gtf_for_genes_in_genome.py --gtf ${gtf} --fasta ${genome} -o genome_filtered_genes.gtf	
gawk '( \$3 ~ /gene/ )' genome_filtered_genes.gtf > new.gtf	
gawk '( \$3 ~ /transcript/ )' genome_filtered_genes.gtf >> new.gtf	
gawk '( \$3 ~ /exon/ && \$7 ~ /+/ )' genome_filtered_genes.gtf | sort -k1,1 -k4,4n >> new.gtf	
gawk '( \$3 ~ /exon/ && \$7 ~ /-/ )' genome_filtered_genes.gtf | sort -k1,1 -k4,4nr >> new.gtf

if [[ \$(awk '{print \$3}' new.gtf | grep -c transcript) -le 1 ]]; then
    echo "transcript entries are not found in gtf file. gffread will add transcript entries."
    gffread -E --keep-genes new.gtf -T -o- >_${gtf} 2>gffread.log
else
    ln -s new.gtf _${gtf}
fi


mkdir -p kallisto_${name}
if [ "${mate}" == "pair" ]; then
    kallisto quant ${kallisto_parameters} -i ${kallisto_index} ${genomebamText} -o kallisto_${name} ${reads}  > kallisto_${name}/kallisto.log 2>&1
else
    kallisto quant --single ${kallisto_parameters} ${fragment_lengthText} ${standard_deviationText}  -i ${kallisto_index} ${genomebamText} -o kallisto_${name} ${reads} > kallisto_${name}/kallisto.log 2>&1
fi


if ls kallisto_${name}/*.bam 1> /dev/null 2>&1; then
    mv kallisto_${name}/*.bam  ${name}.bam
    samtools view -b -F 4  ${name}.bam -o  ${name}_clean.bam	
    samtools sort -o ${name}_sorted.bam ${name}_clean.bam && samtools index ${name}_sorted.bam
    rm ${name}_clean.bam ${name}.bam
fi
if ls kallisto_${name}/*.bam.bai 1> /dev/null 2>&1; then
    rm -r kallisto_${name}/*.bam.bai 
fi

if [ -f kallisto_${name}/abundance.tsv ]; then
   mv kallisto_${name}/abundance.tsv  kallisto_${name}/abundance_isoforms.tsv
fi



"""

}

//* params.gtf =  ""  //* @input

//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 30
    $CPU  = 1
    $MEMORY = 10
    $QUEUE = "short"
}
//* platform
//* autofill

process Kallisto_module_Kallisto_transcript_to_gene_count {

input:
 file outDir from g248_36_outputDir00_g248_38
 file gtf from g245_54_gtfFile01_g248_38

output:
 file newoutDir  into g248_38_outputDir00_g248_39

shell:
newoutDir = "genes_" + outDir
'''
#!/usr/bin/env perl
use strict;
use Getopt::Long;
use IO::File;
use Data::Dumper;

my $gtf_file = "!{gtf}";
my $kallisto_transcript_matrix_in = "!{outDir}/abundance_isoforms.tsv";
my $kallisto_transcript_matrix_out = "!{outDir}/abundance_genes.tsv";
open(IN, "<$gtf_file") or die "Can't open $gtf_file.\\n";
my %all_genes; # save gene_id of transcript_id
while(<IN>){
  next if(/^##/); #ignore header
  chomp;
  my %attribs = ();
  my ($chr, $source, $type, $start, $end, $score,
    $strand, $phase, $attributes) = split("\\t");
  my @add_attributes = split(";", $attributes);
  # store ids and additional information in second hash
  foreach my $attr ( @add_attributes ) {
     next unless $attr =~ /^\\s*(.+)\\s(.+)$/;
     my $c_type  = $1;
     my $c_value = $2;
     $c_value =~ s/\\"//g;
     if($c_type  && $c_value){
       if(!exists($attribs{$c_type})){
         $attribs{$c_type} = [];
       }
       push(@{ $attribs{$c_type} }, $c_value);
     }
  }
  #work with the information from the two hashes...
  if(exists($attribs{'transcript_id'}->[0]) && exists($attribs{'gene_id'}->[0])){
    if(!exists($all_genes{$attribs{'transcript_id'}->[0]})){
        $all_genes{$attribs{'transcript_id'}->[0]} = $attribs{'gene_id'}->[0];
    }
  } 
}


# print Dumper \\%all_genes;

#Parse the kallisto input file, determine gene IDs for each transcript, and calculate sum TPM values
my %gene_exp;
my %gene_length;
my %samples;
my $ki_fh = IO::File->new($kallisto_transcript_matrix_in, 'r');
my $header = '';
my $h = 0;
while (my $ki_line = $ki_fh->getline) {
  $h++;
  chomp($ki_line);
  my @ki_entry = split("\\t", $ki_line);
  my $s = 0;
  if ($h == 1){
    $header = $ki_line;
    my $first_col = shift @ki_entry;
    my $second_col = shift @ki_entry;
    foreach my $sample (@ki_entry){
      $s++;
      $samples{$s}{name} = $sample;
    }
    next;
  }
  my $trans_id = shift @ki_entry;
  my $length = shift @ki_entry;
  my $gene_id;
  if ($all_genes{$trans_id}){
    $gene_id = $all_genes{$trans_id};
  }elsif($trans_id =~ /ERCC/){
    $gene_id = $trans_id;
  }else{
    print "\\n\\nCould not identify gene id from trans id: $trans_id\\n\\n";
  }

  $s = 0;
  foreach my $value (@ki_entry){
    $s++;
    $gene_exp{$gene_id}{$s} += $value;
  }
  if ($gene_length{$gene_id}){
    $gene_length{$gene_id} = $length if ($length > $gene_length{$gene_id});
  }else{
    $gene_length{$gene_id} = $length;
  }

}
$ki_fh->close;

my $ko_fh = IO::File->new($kallisto_transcript_matrix_out, 'w');
unless ($ko_fh) { die('Failed to open file: '. $kallisto_transcript_matrix_out); }

print $ko_fh "$header\\n";
foreach my $gene_id (sort {$a cmp $b} keys %gene_exp){
  print $ko_fh "$gene_id\\t$gene_length{$gene_id}\\t";
  my @vals;
  foreach my $s (sort {$a <=> $b} keys %samples){
     push(@vals, $gene_exp{$gene_id}{$s});
  }
  my $val_string = join("\\t", @vals);
  print $ko_fh "$val_string\\n";
}


$ko_fh->close;
if (checkFile("!{outDir}")){
	rename ("!{outDir}", "!{newoutDir}");
}

sub checkFile {
    my ($file) = @_;
    print "$file\\n";
    return 1 if ( -e $file );
    return 0;
}

'''
}

//* params.gtf =  ""  //* @input

//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 30
    $CPU  = 1
    $MEMORY = 10
    $QUEUE = "short"
}
//* platform
//* autofill

process Kallisto_module_Kallisto_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "kallisto_count/$filename"}
input:
 file kallistoOut from g248_38_outputDir00_g248_39.collect()
 file gtf from g245_54_gtfFile01_g248_39

output:
 file "*.tsv"  into g248_39_outputFile00_g303_25, g248_39_outputFile00_g303_24

shell:
'''
#!/usr/bin/env perl
use Data::Dumper;
use strict;

### Parse gtf file
my $gtf_file = "!{gtf}";
open(IN, "<$gtf_file") or die "Can't open $gtf_file.\\n";
my %all_genes; # save gene_id of transcript_id
my %all_trans; # map transcript_id of genes
while(<IN>){
  next if(/^##/); #ignore header
  chomp;
  my %attribs = ();
  my ($chr, $source, $type, $start, $end, $score,
    $strand, $phase, $attributes) = split("\\t");
  my @add_attributes = split(";", $attributes);
  # store ids and additional information in second hash
  foreach my $attr ( @add_attributes ) {
     next unless $attr =~ /^\\s*(.+)\\s(.+)$/;
     my $c_type  = $1;
     my $c_value = $2;
     $c_value =~ s/\\"//g;
     if($c_type  && $c_value){
       if(!exists($attribs{$c_type})){
         $attribs{$c_type} = [];
       }
       push(@{ $attribs{$c_type} }, $c_value);
     }
  }
  #work with the information from the two hashes...
  if(exists($attribs{'transcript_id'}->[0]) && exists($attribs{'gene_id'}->[0])){
    if(!exists($all_genes{$attribs{'transcript_id'}->[0]})){
        $all_genes{$attribs{'transcript_id'}->[0]} = $attribs{'gene_id'}->[0];
    }
    if(!exists($all_trans{$attribs{'gene_id'}->[0]})){
        $all_trans{$attribs{'gene_id'}->[0]} = $attribs{'transcript_id'}->[0];
    } else {
    	if (index($all_trans{$attribs{'gene_id'}->[0]}, $attribs{'transcript_id'}->[0]) == -1) {
			$all_trans{$attribs{'gene_id'}->[0]} = $all_trans{$attribs{'gene_id'}->[0]} . "," .$attribs{'transcript_id'}->[0];
		}
    	
    }
  } 
}


print Dumper \\%all_trans;



#### Create summary table

my %tf = (
        expected_count => 3,
        tpm => 4
    );

my $indir = $ENV{'PWD'};
my $outdir = $ENV{'PWD'};

my @gene_iso_ar = ("genes","isoforms");
my @tpm_fpkm_expectedCount_ar = ("expected_count", "tpm");
for(my $l = 0; $l <= $#gene_iso_ar; $l++) {
    my $gene_iso = $gene_iso_ar[$l];
    for(my $ll = 0; $ll <= $#tpm_fpkm_expectedCount_ar; $ll++) {
        my $tpm_fpkm_expectedCount = $tpm_fpkm_expectedCount_ar[$ll];

        opendir D, $indir or die "Could not open $indir\\n";
        my @alndirs = sort { $a cmp $b } grep /^genes_kallisto_/, readdir(D);
        closedir D;
    
        my @a=();
        my %b=();
        my %c=();
        my $i=0;
        foreach my $d (@alndirs){ 
            my $dir = "${indir}/$d";
            print $d."\\n";
            my $libname=$d;
            $libname=~s/genes_kallisto_//;
            $i++;
            $a[$i]=$libname;
            open IN,"${dir}/abundance_${gene_iso}.tsv";
            $_=<IN>;
            while(<IN>)
            {
                my @v=split; 
                # $v[0] -> transcript_id
                # $all_genes{$v[0]} -> $gene_id
                if ($gene_iso eq "isoforms"){
                	$c{$v[0]}=$all_genes{$v[0]};
                } elsif ($gene_iso eq "genes"){
                	$c{$v[0]}=$all_trans{$v[0]};
                } 
                $b{$v[0]}{$i}=$v[$tf{$tpm_fpkm_expectedCount}];
                 
            }
            close IN;
        }
        my $outfile="${indir}/"."$gene_iso"."_expression_"."$tpm_fpkm_expectedCount".".tsv";
        open OUT, ">$outfile";
        if ($gene_iso ne "isoforms") {
            print OUT "gene\\ttranscript";
        } else {
            print OUT "transcript\\tgene";
        }
    
        for(my $j=1;$j<=$i;$j++) {
            print OUT "\\t$a[$j]";
        }
        print OUT "\\n";
    
        foreach my $key (keys %b) {
            print OUT "$key\\t$c{$key}";
            for(my $j=1;$j<=$i;$j++){
                print OUT "\\t$b{$key}{$j}";
            }
            print OUT "\\n";
        }
        close OUT;
    }
}

'''
}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_Kallisto_Prepare_DESeq2 {

input:
 file counts from g248_39_outputFile00_g303_24
 file groups_file from g_295_1_g303_24
 file compare_file from g_294_2_g303_24
 val run_DESeq2 from g_310_3_g303_24

output:
 file "DE_reports"  into g303_24_outputFile00_g303_37
 val "_des"  into g303_24_postfix10_g303_33
 file "DE_reports/outputs/*_all_deseq2_results.tsv"  into g303_24_outputFile21_g303_33

container 'quay.io/viascientific/de_module:4.0'

when:
run_DESeq2 == 'yes'

script:

feature_type = params.DE_module_Kallisto_Prepare_DESeq2.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_Kallisto_Prepare_DESeq2.include_distribution
include_all2all = params.DE_module_Kallisto_Prepare_DESeq2.include_all2all
include_pca = params.DE_module_Kallisto_Prepare_DESeq2.include_pca

filter_type = params.DE_module_Kallisto_Prepare_DESeq2.filter_type
min_count = params.DE_module_Kallisto_Prepare_DESeq2.min_count
min_samples = params.DE_module_Kallisto_Prepare_DESeq2.min_samples
min_counts_per_sample = params.DE_module_Kallisto_Prepare_DESeq2.min_counts_per_sample
excluded_events = params.DE_module_Kallisto_Prepare_DESeq2.excluded_events

include_batch_correction = params.DE_module_Kallisto_Prepare_DESeq2.include_batch_correction
batch_correction_column = params.DE_module_Kallisto_Prepare_DESeq2.batch_correction_column
batch_correction_group_column = params.DE_module_Kallisto_Prepare_DESeq2.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_Kallisto_Prepare_DESeq2.batch_normalization_algorithm

transformation = params.DE_module_Kallisto_Prepare_DESeq2.transformation
pca_color = params.DE_module_Kallisto_Prepare_DESeq2.pca_color
pca_shape = params.DE_module_Kallisto_Prepare_DESeq2.pca_shape
pca_fill = params.DE_module_Kallisto_Prepare_DESeq2.pca_fill
pca_transparency = params.DE_module_Kallisto_Prepare_DESeq2.pca_transparency
pca_label = params.DE_module_Kallisto_Prepare_DESeq2.pca_label

include_deseq2 = params.DE_module_Kallisto_Prepare_DESeq2.include_deseq2
input_mode = params.DE_module_Kallisto_Prepare_DESeq2.input_mode
design = params.DE_module_Kallisto_Prepare_DESeq2.design
fitType = params.DE_module_Kallisto_Prepare_DESeq2.fitType
use_batch_corrected_in_DE = params.DE_module_Kallisto_Prepare_DESeq2.use_batch_corrected_in_DE
apply_shrinkage = params.DE_module_Kallisto_Prepare_DESeq2.apply_shrinkage
shrinkage_type = params.DE_module_Kallisto_Prepare_DESeq2.shrinkage_type
include_volcano = params.DE_module_Kallisto_Prepare_DESeq2.include_volcano
include_ma = params.DE_module_Kallisto_Prepare_DESeq2.include_ma
include_heatmap = params.DE_module_Kallisto_Prepare_DESeq2.include_heatmap

padj_significance_cutoff = params.DE_module_Kallisto_Prepare_DESeq2.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_Kallisto_Prepare_DESeq2.fc_significance_cutoff
padj_floor = params.DE_module_Kallisto_Prepare_DESeq2.padj_floor
fc_ceiling = params.DE_module_Kallisto_Prepare_DESeq2.fc_ceiling

convert_names = params.DE_module_Kallisto_Prepare_DESeq2.convert_names
count_file_names = params.DE_module_Kallisto_Prepare_DESeq2.count_file_names
converted_name = params.DE_module_Kallisto_Prepare_DESeq2.converted_name
org_db = params.DE_module_Kallisto_Prepare_DESeq2.org_db
num_labeled = params.DE_module_Kallisto_Prepare_DESeq2.num_labeled
highlighted_genes = params.DE_module_Kallisto_Prepare_DESeq2.highlighted_genes
include_volcano_highlighted = params.DE_module_Kallisto_Prepare_DESeq2.include_volcano_highlighted
include_ma_highlighted = params.DE_module_Kallisto_Prepare_DESeq2.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name, org_db},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm, use_batch_corrected_in_DE},{include_batch_correction="false"},{include_deseq2="true", design, fitType, apply_shrinkage, shrinkage_type, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, org_db, num_labeled, highlighted_genes},{include_deseq2="false"},{apply_shrinkage="true", shrinkage_type},{apply_shrinkage="false"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples, min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{design, fitType, use_batch_corrected_in_DE, apply_shrinkage, shrinkage_type},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name, org_db},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_deseq2 = include_deseq2 == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
apply_shrinkage = apply_shrinkage == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir reports
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_DESeq2.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-DESeq2 ${include_deseq2} --input-mode ${input_mode} --design '${design}' --fitType ${fitType} --use-batch-correction-in-DE ${use_batch_corrected_in_DE} --apply-shrinkage ${apply_shrinkage} --shrinkage-type ${shrinkage_type} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --org-db ${org_db} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_Kallisto_Prepare_LimmaVoom {

input:
 file counts from g248_39_outputFile00_g303_25
 file groups_file from g_295_1_g303_25
 file compare_file from g_294_2_g303_25
 val run_limmaVoom from g_360_3_g303_25

output:
 file "DE_reports"  into g303_25_outputFile00_g303_39
 val "_lv"  into g303_25_postfix10_g303_41
 file "DE_reports/outputs/*_all_limmaVoom_results.tsv"  into g303_25_outputFile21_g303_41

container 'quay.io/viascientific/de_module:4.0'

when:
run_limmaVoom == 'yes'

script:

feature_type = params.DE_module_Kallisto_Prepare_LimmaVoom.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_Kallisto_Prepare_LimmaVoom.include_distribution
include_all2all = params.DE_module_Kallisto_Prepare_LimmaVoom.include_all2all
include_pca = params.DE_module_Kallisto_Prepare_LimmaVoom.include_pca

filter_type = params.DE_module_Kallisto_Prepare_LimmaVoom.filter_type
min_count = params.DE_module_Kallisto_Prepare_LimmaVoom.min_count
min_samples = params.DE_module_Kallisto_Prepare_LimmaVoom.min_samples
min_counts_per_sample = params.DE_module_Kallisto_Prepare_LimmaVoom.min_counts_per_sample
excluded_events = params.DE_module_Kallisto_Prepare_LimmaVoom.excluded_events

include_batch_correction = params.DE_module_Kallisto_Prepare_LimmaVoom.include_batch_correction
batch_correction_column = params.DE_module_Kallisto_Prepare_LimmaVoom.batch_correction_column
batch_correction_group_column = params.DE_module_Kallisto_Prepare_LimmaVoom.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_Kallisto_Prepare_LimmaVoom.batch_normalization_algorithm

transformation = params.DE_module_Kallisto_Prepare_LimmaVoom.transformation
pca_color = params.DE_module_Kallisto_Prepare_LimmaVoom.pca_color
pca_shape = params.DE_module_Kallisto_Prepare_LimmaVoom.pca_shape
pca_fill = params.DE_module_Kallisto_Prepare_LimmaVoom.pca_fill
pca_transparency = params.DE_module_Kallisto_Prepare_LimmaVoom.pca_transparency
pca_label = params.DE_module_Kallisto_Prepare_LimmaVoom.pca_label

include_limma = params.DE_module_Kallisto_Prepare_LimmaVoom.include_limma
use_batch_corrected_in_DE = params.DE_module_Kallisto_Prepare_LimmaVoom.use_batch_corrected_in_DE
normalization_method = params.DE_module_Kallisto_Prepare_LimmaVoom.normalization_method
logratioTrim = params.DE_module_Kallisto_Prepare_LimmaVoom.logratioTrim
sumTrim = params.DE_module_Kallisto_Prepare_LimmaVoom.sumTrim
Acutoff = params.DE_module_Kallisto_Prepare_LimmaVoom.Acutoff
doWeighting = params.DE_module_Kallisto_Prepare_LimmaVoom.doWeighting
p = params.DE_module_Kallisto_Prepare_LimmaVoom.p
include_volcano = params.DE_module_Kallisto_Prepare_LimmaVoom.include_volcano
include_ma = params.DE_module_Kallisto_Prepare_LimmaVoom.include_ma
include_heatmap = params.DE_module_Kallisto_Prepare_LimmaVoom.include_heatmap

padj_significance_cutoff = params.DE_module_Kallisto_Prepare_LimmaVoom.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_Kallisto_Prepare_LimmaVoom.fc_significance_cutoff
padj_floor = params.DE_module_Kallisto_Prepare_LimmaVoom.padj_floor
fc_ceiling = params.DE_module_Kallisto_Prepare_LimmaVoom.fc_ceiling

convert_names = params.DE_module_Kallisto_Prepare_LimmaVoom.convert_names
count_file_names = params.DE_module_Kallisto_Prepare_LimmaVoom.count_file_names
converted_name = params.DE_module_Kallisto_Prepare_LimmaVoom.converted_name
num_labeled = params.DE_module_Kallisto_Prepare_LimmaVoom.num_labeled
highlighted_genes = params.DE_module_Kallisto_Prepare_LimmaVoom.highlighted_genes
include_volcano_highlighted = params.DE_module_Kallisto_Prepare_LimmaVoom.include_volcano_highlighted
include_ma_highlighted = params.DE_module_Kallisto_Prepare_LimmaVoom.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm,use_batch_corrected_in_DE},{include_batch_correction="false"},{include_limma="true", normalization_method, logratioTrim, sumTrim, doWeighting, Acutoff, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, num_labeled, highlighted_genes},{include_limma="false"},{normalization_method="TMM", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="TMMwsp", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="RLE"},{normalization_method="upperquartile", p},{normalization_method="none"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples,min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_limma, use_batch_corrected_in_DE},{normalization_method,logratioTrim,sumTrim,doWeighting,Acutoff,p},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_limma = include_limma == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

doWeighting = doWeighting == 'true' ? 'TRUE' : 'FALSE'
TMM_args = normalization_method.equals('TMM') || normalization_method.equals('TMMwsp') ? '--logratio-trim ' + logratioTrim + ' --sum-trim ' + sumTrim + ' --do-weighting ' + doWeighting + ' --A-cutoff="' + Acutoff + '"' : ''
upperquartile_args = normalization_method.equals('upperquartile') ? '--p ' + p : ''

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_limmaVoom.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-limma ${include_limma} \
--use-batch-correction-in-DE ${use_batch_corrected_in_DE} --normalization-method ${normalization_method} ${TMM_args} ${upperquartile_args} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}

//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 30
    $CPU  = 1
    $MEMORY = 10
    $QUEUE = "short"
}
//* platform
//* autofill

process Kallisto_module_Kallisto_Alignment_Summary {

input:
 file kallistoDir from g248_36_outputDir00_g248_22.collect()

output:
 file "kallisto_alignment_sum.tsv"  into g248_22_outFileTSV09_g_198

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;
my $indir = $ENV{'PWD'};

opendir D, $indir or die "Could not open $indir";
my @alndirs = sort { $a cmp $b } grep /^kallisto_/, readdir(D);
closedir D;

my @a=();
my %b=();
my %c=();
my $i=0;
my @headers = ();
my %tsv;
foreach my $d (@alndirs){
    my $dir = "${indir}/$d";
    my $libname=$d;
    $libname=~s/kallisto_//;
    my $multimapped;
    my $aligned;
    my $total;
    my $unique;
    
    # eg. [quant] processed 24,788 reads, 19,238 reads pseudoaligned
	chomp($total   = `cat ${dir}/kallisto.log | grep 'pseudoaligned' | sed 's/,//g' | awk '{sum+=\\$3} END {print sum}'`);
	chomp($aligned = `cat ${dir}/kallisto.log | grep 'pseudoaligned' | sed 's/,//g' | awk '{sum+=\\$5} END {print sum}'`);
	chomp($unique = `cat ${dir}/run_info.json | grep 'n_unique' | sed 's/,//g' | sed 's/"//g' | sed 's/://g' | awk '{sum+=\\$2} END {print sum}'`);
    $tsv{$libname}=[$libname, $total];
    push(@{$tsv{$libname}}, $aligned);
    push(@{$tsv{$libname}}, $unique);
}

push(@headers, "Sample");
push(@headers, "Total Reads");
push(@headers, "Pseudoaligned Reads (Kallisto)");
push(@headers, "Uniquely Mapped Reads (Kallisto)");

my @keys = keys %tsv;
my $summary = "kallisto_alignment_sum.tsv";
my $header_string = join("\\t", @headers);
`echo "$header_string" > $summary`;
foreach my $key (@keys){
    my $values = join("\\t", @{ $tsv{$key} });
        `echo "$values" >> $summary`;
}
'''
}


process BAM_Analysis_Module_Kallisto_bam_sort_index {

input:
 set val(name), file(bam) from g248_36_bam_file10_g255_143

output:
 set val(name), file("bam/*.bam"), file("bam/*.bam.bai")  into g255_143_bam_bai00_g255_134, g255_143_bam_bai00_g255_142

when:
params.run_BigWig_Conversion == "yes" || params.run_RSeQC == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}

"""
$runSamtools
mkdir -p bam
mv ${name}_sorted.bam ${name}_sorted.bam.bai bam/.
"""
}



//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 30
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_Kallisto_UCSC_BAM2BigWig_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.bw$/) "bigwig_kallisto/$filename"}
input:
 set val(name), file(bam), file(bai) from g255_143_bam_bai00_g255_142
 file genomeSizes from g245_54_genomeSizes21_g255_142

output:
 file "*.bw" optional true  into g255_142_outputFileBw00
 file "publish/*.bw" optional true  into g255_142_publishBw10_g255_145

container 'quay.io/biocontainers/deeptools:3.5.4--pyhdfd78af_1'

when:
params.run_BigWig_Conversion == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = ""
    nameFinal = nameAll
} else {
    runSamtools = "mv $bam ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
deeptools_parameters = params.BAM_Analysis_Module_Kallisto_UCSC_BAM2BigWig_converter.deeptools_parameters
visualize_bigwig_in_reports = params.BAM_Analysis_Module_Kallisto_UCSC_BAM2BigWig_converter.visualize_bigwig_in_reports

"""
$runSamtools
bamCoverage ${deeptools_parameters}  -b ${nameFinal} -o ${name}.bw 

if [ "${visualize_bigwig_in_reports}" == "yes" ]; then
	mkdir -p publish
	mv ${name}.bw publish/.
fi
"""

}



process BAM_Analysis_Module_Kallisto_Genome_Browser {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "genome_browser_kallisto/$filename"}
input:
 file bigwigs from g255_142_publishBw10_g255_145.collect()
 file group_file from g_295_1_g255_145

output:
 file "*"  into g255_145_bigWig_file00

container 'quay.io/viascientific/python-basics:2.0'


script:
try {
    myVariable = group_file
} catch (MissingPropertyException e) {
    group_file = ""
}

genome_build_short= ""
if (params.genome_build == "mousetest_mm10"){
    genome_build_short= "mm10"
} else if (params.genome_build == "human_hg19_refseq"){
    genome_build_short = "hg19"
} else if (params.genome_build == "human_hg38_gencode_v28"){
    genome_build_short = "hg38"
} else if (params.genome_build == "human_hg38_gencode_v34"){
    genome_build_short = "hg38"
} else if (params.genome_build == "mouse_mm10_refseq"){
    genome_build_short = "mm10"
} else if (params.genome_build == "mouse_mm10_gencode_m25"){
    genome_build_short = "mm10"
} else if (params.genome_build == "rat_rn6_refseq"){
    genome_build_short = "rn6"
} else if (params.genome_build == "rat_rn6_ensembl_v86"){
    genome_build_short = "rn6"
} else if (params.genome_build == "zebrafish_GRCz11_ensembl_v95"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_refseq"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_v4.3.2"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "c_elegans_ce11_ensembl_ws245"){
    genome_build_short = "ce11"
} else if (params.genome_build == "d_melanogaster_dm6_refseq"){
    genome_build_short = "dm6"
} else if (params.genome_build == "s_cerevisiae_sacCer3_refseq"){
    genome_build_short = "sacCer3"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v31"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v51"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "e_coli_ASM584v2_refseq"){
    genome_build_short = "ASM584v2"
} else if (params.genome_build == "dog_canFam3_refseq"){
    genome_build_short = "canFam3"
} 

"""

#!/usr/bin/env python

import glob,json,os,sys,csv,random,shutil,requests,subprocess

def check_url_existence(url):
    try:
        response = requests.head(url, allow_redirects=True)
        # Check if the status code is in the success range (200-399)
        return 200 <= response.status_code < 400
    except requests.ConnectionError:
        return False

def get_lib_val(str, lib):
    for key, value in lib.items():
        if key.lower() in str.lower():
            return value
    return None

def find_and_move_folders_with_bw_files(start_dir):
    bigwig_dir = "bigwigs"
    if os.path.exists(bigwig_dir)!=True:
        os.makedirs(bigwig_dir)
    subprocess.getoutput("cp ${bigwigs} bigwigs/. ")

def Generate_HubFile(groupfile):
    # hub.txt
    Hub = open("hub.txt", "w")
    Hub.write("hub UCSCHub \\n")
    Hub.write("shortLabel UCSCHub \\n")
    Hub.write("longLabel UCSCHub \\n")
    Hub.write("genomesFile genomes.txt \\n")
    Hub.write("email support@viascientific.com \\n")
    Hub.write("\\n")
    Hub.close()

    # genomes.txt
    genomes = open("genomes.txt", "w")
    genomes.write("genome ${genome_build_short} \\n")
    genomes.write("trackDb bigwigs/trackDb.txt \\n")
    genomes.write("\\n")
    genomes.close()
    #trackDb = open("bigwigs/trackDb.txt", "w")
    path = r'bigwigs/*.bw'
    files = glob.glob(path)
    files.sort()
    sample={}
    for i in files:
        temp=i.split('.')[0]
        temp=temp.replace('bigwigs/','')
        if temp in sample.keys():
            sample[temp].append(i.replace('bigwigs/',''))
        else:
            sample[temp]=[]
            sample[temp].append(i.replace('bigwigs/',''))
    second_layer_indent=" "*2
    third_layer_indent = " " * 14
    if groupfile == "" or os.stat(groupfile).st_size == 0:
        #No GroupFile
        trackDb = open("bigwigs/trackDb.txt", "w")
        for i in sample.keys():
            trackDb.write('track %s\\n' %i)
            trackDb.write('shortLabel %s\\n' %i)
            trackDb.write('longLabel %s\\n' %i)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for j in sample[i]:
                trackDb.write(second_layer_indent+'track %s\\n' % j)
                trackDb.write(second_layer_indent+'bigDataUrl %s\\n' % j)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %i)
                trackDb.write('\\n')
        trackDb.close()
    else:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_to_sample={}
        for cond in condition:
            if cond not in condition_to_sample.keys():
                condition_to_sample[cond]=[]
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
            else:
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
        trackDb = open("bigwigs/trackDb.txt", "w")
        for cond in condition:
            trackDb.write('track %s\\n' %cond)
            trackDb.write('shortLabel %s\\n' %cond)
            trackDb.write('longLabel %s\\n' %cond)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for samp in condition_to_sample[cond]:
                trackDb.write(second_layer_indent+'track %s\\n' % samp)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'view Signal \\n')
                trackDb.write(second_layer_indent+'visibility full \\n')
                trackDb.write(second_layer_indent+'viewLimits 0:20 \\n')
                trackDb.write(second_layer_indent+'autoScale on \\n')
                trackDb.write(second_layer_indent+'maxHeightPixels 128:20:8 \\n')
                trackDb.write(second_layer_indent+'configurable on \\n')
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %cond)
                trackDb.write('\\n')
                for file in sample[samp]:
                    trackDb.write(third_layer_indent + 'track %s\\n' % file)
                    trackDb.write(third_layer_indent + 'bigDataUrl %s\\n' % file)
                    trackDb.write(third_layer_indent + 'shortLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'longLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'type bigWig\\n')
                    trackDb.write(third_layer_indent+'parent %s\\n' %samp)

                    trackDb.write('\\n')

        trackDb.close()


def Generating_Json_files(groupfile):
    publishWebDir = '{{DNEXT_WEB_REPORT_DIR}}/' + 'genome_browser_kallisto' + "/bigwigs"
    locusLib = {}
    # MYC locations
    locusLib["hg19"] = "chr8:128,746,315-128,755,680"
    locusLib["hg38"] = "chr8:127,733,434-127,744,951"
    locusLib["mm10"] = "chr15:61,983,341-61,992,361"
    locusLib["rn6"] = "chr7:102,584,313-102,593,240"
    locusLib["dm6"] = "chrX:3,371,159-3,393,697"
    locusLib["canFam3"] = "chr13:25,198,772-25,207,309"

    cytobandLib = {}
    cytobandLib["hg19"] = "https://igv-genepattern-org.s3.amazonaws.com/genomes/seq/hg19/cytoBand.txt"
    cytobandLib["hg38"] = "https://s3.amazonaws.com/igv.org.genomes/hg38/annotations/cytoBandIdeo.txt.gz"
    cytobandLib["mm10"] = "https://s3.amazonaws.com/igv.broadinstitute.org/annotations/mm10/cytoBandIdeo.txt.gz"
    cytobandLib["rn6"] = "https://s3.amazonaws.com/igv.org.genomes/rn6/cytoBand.txt.gz"
    cytobandLib["dm6"] = "https://s3.amazonaws.com/igv.org.genomes/dm6/cytoBandIdeo.txt.gz"
    cytobandLib["ce11"] = "https://s3.amazonaws.com/igv.org.genomes/ce11/cytoBandIdeo.txt.gz"
    cytobandLib["canFam3"] = "https://s3.amazonaws.com/igv.org.genomes/canFam3/cytoBandIdeo.txt.gz"

    # Get the basename of the original path
    gtf_source_base_name = os.path.basename("${params.gtf_source}")
    gtf_source_sorted = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz")
    print(gtf_source_sorted)
    gtf_source_sorted_index = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz.tbi")
    print(gtf_source_sorted_index)


    data = {}
    data["reference"] = {}
    data["reference"]["id"] = "${params.genome_build}"
    data["reference"]["name"] = "${params.genome_build}"
    data["reference"]["fastaURL"] = "${params.genome_source}"
    data["reference"]["indexURL"] = "${params.genome_source}.fai"
    cytobandurl = get_lib_val("${params.genome_build}", cytobandLib)
    locusStr = get_lib_val("${params.genome_build}", locusLib)
    if cytobandurl is not None:
        data["reference"]["cytobandURL"] = cytobandurl
    if locusStr is not None:
        data["locus"] = []
        data["locus"].append(locusStr)
    data["tracks"] = []
    # prepare gtf Track
    gtfTrack = {}
    gtfTrack["name"] = "${params.genome_build}"
    gtfTrack["gtf"] = "gtf"
    if check_url_existence(gtf_source_sorted):
        gtfTrack["url"] = gtf_source_sorted
    else:
        gtfTrack["url"] = "${params.gtf_source}"
    if check_url_existence(gtf_source_sorted_index):
        gtfTrack["indexURL"] = gtf_source_sorted_index

    # prepare cytobands Track
    if groupfile and os.path.isfile(groupfile) and os.stat(groupfile).st_size != 0:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_color_dict = dict()
        for cond in condition:
            r = lambda: random.randint(0, 255)
            condition_color_dict[cond] = '#%02X%02X%02X' % (r(), r(), r())

    # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                if basename in data_parsed:
                    newdata[file] = data_parsed[basename]
                    newdata[file]["fullname"] = file

        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": condition_color_dict[newdata[j]['group']],
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()
    else:

        # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                newdata[file] = basename
                newdata[file] = file
        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            r = lambda: random.randint(0, 255)
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": '#%02X%02X%02X' % (r(), r(), r()),
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()

if __name__ == "__main__":
    find_and_move_folders_with_bw_files(".")
    Generate_HubFile(groupfile="${group_file}")
    Generating_Json_files(groupfile="${group_file}")

"""
}

//* params.bed =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 10
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_Kallisto_RSeQC {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "rseqc_kallisto/$filename"}
input:
 set val(name), file(bam), file(bai) from g255_143_bam_bai00_g255_134
 file bed from g245_54_bed31_g255_134
 val mate from g_347_mate12_g255_134

output:
 file "*"  into g255_134_outputFileOut014_g_177

container 'quay.io/viascientific/rseqc:1.0'

when:
(params.run_RSeQC && (params.run_RSeQC == "yes")) || !params.run_RSeQC

script:
run_bam_stat = params.BAM_Analysis_Module_Kallisto_RSeQC.run_bam_stat
run_read_distribution = params.BAM_Analysis_Module_Kallisto_RSeQC.run_read_distribution
run_inner_distance = params.BAM_Analysis_Module_Kallisto_RSeQC.run_inner_distance
run_junction_annotation = params.BAM_Analysis_Module_Kallisto_RSeQC.run_junction_annotation
run_junction_saturation = params.BAM_Analysis_Module_Kallisto_RSeQC.run_junction_saturation
//run_geneBody_coverage and run_infer_experiment needs subsampling
run_geneBody_coverage = params.BAM_Analysis_Module_Kallisto_RSeQC.run_geneBody_coverage
run_infer_experiment = params.BAM_Analysis_Module_Kallisto_RSeQC.run_infer_experiment
"""
if [ "$run_bam_stat" == "true" ]; then bam_stat.py  -i ${bam} > ${name}.bam_stat.txt; fi
if [ "$run_read_distribution" == "true" ]; then read_distribution.py  -i ${bam} -r ${bed}> ${name}.read_distribution.out; fi


if [ "$run_infer_experiment" == "true" -o "$run_geneBody_coverage" == "true" ]; then
	numAlignedReads=\$(samtools view -c -F 4 $bam)

	if [ "\$numAlignedReads" -gt 1000000 ]; then
    	echo "Read number is greater than 1000000. Subsampling..."
    	finalRead=1000000
    	fraction=\$(samtools idxstats  $bam | cut -f3 | awk -v ct=\$finalRead 'BEGIN {total=0} {total += \$1} END {print ct/total}')
    	samtools view -b -s \${fraction} $bam > ${name}_sampled.bam
    	samtools index ${name}_sampled.bam
    	if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i ${name}_sampled.bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i ${name}_sampled.bam  -r $bed -o ${name}; fi
	else
		if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i $bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i $bam  -r $bed -o ${name}; fi
	fi

fi


if [ "${mate}" == "pair" ]; then
	if [ "$run_inner_distance" == "true" ]; then inner_distance.py -i $bam  -r $bed -o ${name}.inner_distance > stdout.txt; fi
	if [ "$run_inner_distance" == "true" ]; then head -n 2 stdout.txt > ${name}.inner_distance_mean.txt; fi
fi
if [ "$run_junction_annotation" == "true" ]; then junction_annotation.py -i $bam  -r $bed -o ${name}.junction_annotation 2> ${name}.junction_annotation.log; fi
if [ "$run_junction_saturation" == "true" ]; then junction_saturation.py -i $bam  -r $bed -o ${name}; fi
if [ -e class.log ] ; then mv class.log ${name}_class.log; fi
if [ -e log.txt ] ; then mv log.txt ${name}_log.txt; fi
if [ -e stdout.txt ] ; then mv stdout.txt ${name}_stdout.txt; fi


"""

}

//* params.pdfbox_path =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 32
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 240
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "short"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 500
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
}
//* platform
//* autofill

process BAM_Analysis_Module_Kallisto_Picard {

input:
 set val(name), file(bam) from g248_36_bam_file10_g255_121

output:
 file "*_metrics"  into g255_121_outputFileOut00_g255_82
 file "results/*.pdf"  into g255_121_outputFilePdf12_g255_82

container 'quay.io/viascientific/picard:1.0'

when:
(params.run_Picard_CollectMultipleMetrics && (params.run_Picard_CollectMultipleMetrics == "yes")) || !params.run_Picard_CollectMultipleMetrics

script:
"""
picard CollectMultipleMetrics OUTPUT=${name}_multiple.out VALIDATION_STRINGENCY=LENIENT INPUT=${bam}
mkdir results && java -jar ${params.pdfbox_path} PDFMerger *.pdf results/${name}_multi_metrics.pdf
"""
}


process BAM_Analysis_Module_Kallisto_Picard_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "picard_summary_kallisto/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "rseqc_summary_kallisto/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /results\/.*.pdf$/) "picard_summary_pdf_kallisto/$filename"}
input:
 file picardOut from g255_121_outputFileOut00_g255_82.collect()
 val mate from g_347_mate11_g255_82
 file picardPdf from g255_121_outputFilePdf12_g255_82.collect()

output:
 file "*.tsv"  into g255_82_outputFileTSV00
 file "results/*.pdf"  into g255_82_outputFilePdf11

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage; 
use Data::Dumper;

runCommand("mkdir results && mv *.pdf results/. ");

my $indir = $ENV{'PWD'};
my $outd = $ENV{'PWD'};
my @files = ();
my @outtypes = ("CollectRnaSeqMetrics", "alignment_summary_metrics", "base_distribution_by_cycle_metrics", "insert_size_metrics", "quality_by_cycle_metrics", "quality_distribution_metrics" );

foreach my $outtype (@outtypes)
{
my $ext="_multiple.out";
$ext.=".$outtype" if ($outtype ne "CollectRnaSeqMetrics");
@files = <$indir/*$ext>;

my @rowheaders=();
my @libs=();
my %metricvals=();
my %histvals=();

my $pdffile="";
my $libname="";
foreach my $d (@files){
  my $libname=basename($d, $ext);
  print $libname."\\n";
  push(@libs, $libname); 
  getMetricVals($d, $libname, \\%metricvals, \\%histvals, \\@rowheaders);
}

my $sizemetrics = keys %metricvals;
write_results("$outd/$outtype.stats.tsv", \\@libs,\\%metricvals, \\@rowheaders, "metric") if ($sizemetrics>0);
my $sizehist = keys %histvals;
write_results("$outd/$outtype.hist.tsv", \\@libs,\\%histvals, "none", "nt") if ($sizehist>0);

}

sub write_results
{
  my ($outfile, $libs, $vals, $rowheaders, $name )=@_;
  open(OUT, ">$outfile");
  print OUT "$name\\t".join("\\t", @{$libs})."\\n";
  my $size=0;
  $size=scalar(@{${$vals}{${$libs}[0]}}) if(exists ${$libs}[0] and exists ${$vals}{${$libs}[0]} );
  
  for (my $i=0; $i<$size;$i++)
  { 
    my $rowname=$i;
    $rowname = ${$rowheaders}[$i] if ($name=~/metric/);
    print OUT $rowname;
    foreach my $lib (@{$libs})
    {
      print OUT "\\t".${${$vals}{$lib}}[$i];
    } 
    print OUT "\\n";
  }
  close(OUT);
}

sub getMetricVals{
  my ($filename, $libname, $metricvals, $histvals,$rowheaders)=@_;
  if (-e $filename){
     my $nextisheader=0;
     my $nextisvals=0;
     my $nexthist=0;
     open(IN, $filename);
     while(my $line=<IN>)
     {
       chomp($line);
       @{$rowheaders}=split(/\\t/, $line) if ($nextisheader && !scalar(@{$rowheaders})); 
       if ($nextisvals) {
         @{${$metricvals}{$libname}}=split(/\\t/, $line);
         $nextisvals=0;
       }
       if($nexthist){
          my @vals=split(/[\\s\\t]+/,$line); 
          push(@{${$histvals}{$libname}}, $vals[1]) if (exists $vals[1]);
       }
       $nextisvals=1 if ($nextisheader); $nextisheader=0;
       $nextisheader=1 if ($line=~/METRICS CLASS/);
       $nexthist=1 if ($line=~/normalized_position/);
     } 
  }
  
}


sub runCommand {
	my ($com) = @_;
	if ($com eq ""){
		return "";
    }
    my $error = system(@_);
	if   ($error) { die "Command failed: $error $com\\n"; }
    else          { print "Command successful: $com\\n"; }
}
'''

}

igv_extention_factor = params.BAM_Analysis_Module_Kallisto_IGV_BAM2TDF_converter.igv_extention_factor
igv_window_size = params.BAM_Analysis_Module_Kallisto_IGV_BAM2TDF_converter.igv_window_size

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 24
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} 
//* platform
//* autofill

process BAM_Analysis_Module_Kallisto_IGV_BAM2TDF_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tdf$/) "igvtools_kallisto/$filename"}
input:
 val mate from g_347_mate10_g255_131
 set val(name), file(bam) from g248_36_bam_file11_g255_131
 file genomeSizes from g245_54_genomeSizes22_g255_131

output:
 file "*.tdf"  into g255_131_outputFileOut00

when:
(params.run_IGV_TDF_Conversion && (params.run_IGV_TDF_Conversion == "yes")) || !params.run_IGV_TDF_Conversion

script:
pairedText = (params.nucleicAcidType == "dna" && mate == "pair") ? " --pairs " : ""
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
"""
$runSamtools
igvtools count -w ${igv_window_size} -e ${igv_extention_factor} ${pairedText} ${nameFinal} ${name}.tdf ${genomeSizes}
"""
}

//* params.hisat2_index =  ""  //* @input


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 4
    $MEMORY = 32
}
//* platform
//* platform
//* autofill

process HISAT2_Module_Map_HISAT2 {

input:
 val mate from g_347_mate10_g249_14
 set val(name), file(reads) from g256_46_reads01_g249_14
 file hisat2index from g249_15_hisat2Index02_g249_14

output:
 set val(name), file("*.bam")  into g249_14_mapped_reads00_g249_13
 set val(name), file("*.align_summary.txt")  into g249_14_outputFileTxt10_g249_2
 set val(name), file("*.flagstat.txt")  into g249_14_outputFileOut22

when:
(params.run_HISAT2 && (params.run_HISAT2 == "yes")) || !params.run_HISAT2

script:
HISAT2_parameters = params.HISAT2_Module_Map_HISAT2.HISAT2_parameters
nameAll = reads.toString()
nameArray = nameAll.split(' ')
file2 = ""
if (nameAll.contains('.gz')) {
    file1 =  nameArray[0]
    if (mate == "pair") {file2 =  nameArray[1] }
} 

"""
basename=\$(basename ${hisat2index}/*.8.ht2 | cut -d. -f1)
if [ "${mate}" == "pair" ]; then
    hisat2 ${HISAT2_parameters} -x ${hisat2index}/\${basename} -1 ${file1} -2 ${file2} -S ${name}.sam &> ${name}.align_summary.txt
else
    hisat2 ${HISAT2_parameters} -x ${hisat2index}/\${basename} -U ${file1} -S ${name}.sam &> ${name}.align_summary.txt
fi
samtools view -bS ${name}.sam > ${name}.bam
rm ${name}.sam
samtools flagstat ${name}.bam > ${name}.flagstat.txt
"""

}


//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 2000
    $CPU  = 1
    $MEMORY = 8
    $QUEUE = "long"
}
//* platform
//* autofill

process HISAT2_Module_Merge_Bam {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_sorted.*bai$/) "hisat2/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_sorted.*bam$/) "hisat2/$filename"}
input:
 set val(oldname), file(bamfiles) from g249_14_mapped_reads00_g249_13.groupTuple()

output:
 set val(oldname), file("${oldname}.bam")  into g249_13_merged_bams00
 set val(oldname), file("*_sorted*bai")  into g249_13_bam_index11
 set val(oldname), file("*_sorted*bam")  into g249_13_sorted_bam20_g280_1, g249_13_sorted_bam21_g252_131, g249_13_sorted_bam20_g252_121, g249_13_sorted_bam20_g252_143

shell:
'''
num=$(echo "!{bamfiles.join(" ")}" | awk -F" " '{print NF-1}')
if [ "${num}" -gt 0 ]; then
    samtools merge !{oldname}.bam !{bamfiles.join(" ")} && samtools sort -o !{oldname}_sorted.bam !{oldname}.bam && samtools index !{oldname}_sorted.bam
else
    mv !{bamfiles.join(" ")} !{oldname}.bam 2>/dev/null || true
    samtools sort  -o !{oldname}_sorted.bam !{oldname}.bam && samtools index !{oldname}_sorted.bam
fi
'''
}


process BAM_Analysis_Module_HISAT2_bam_sort_index {

input:
 set val(name), file(bam) from g249_13_sorted_bam20_g252_143

output:
 set val(name), file("bam/*.bam"), file("bam/*.bam.bai")  into g252_143_bam_bai00_g252_134, g252_143_bam_bai00_g252_142

when:
params.run_BigWig_Conversion == "yes" || params.run_RSeQC == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}

"""
$runSamtools
mkdir -p bam
mv ${name}_sorted.bam ${name}_sorted.bam.bai bam/.
"""
}



//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 30
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_HISAT2_UCSC_BAM2BigWig_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.bw$/) "bigwig_hisat2/$filename"}
input:
 set val(name), file(bam), file(bai) from g252_143_bam_bai00_g252_142
 file genomeSizes from g245_54_genomeSizes21_g252_142

output:
 file "*.bw" optional true  into g252_142_outputFileBw00
 file "publish/*.bw" optional true  into g252_142_publishBw10_g252_145

container 'quay.io/biocontainers/deeptools:3.5.4--pyhdfd78af_1'

when:
params.run_BigWig_Conversion == "yes"

script:
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = ""
    nameFinal = nameAll
} else {
    runSamtools = "mv $bam ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
deeptools_parameters = params.BAM_Analysis_Module_HISAT2_UCSC_BAM2BigWig_converter.deeptools_parameters
visualize_bigwig_in_reports = params.BAM_Analysis_Module_HISAT2_UCSC_BAM2BigWig_converter.visualize_bigwig_in_reports

"""
$runSamtools
bamCoverage ${deeptools_parameters}  -b ${nameFinal} -o ${name}.bw 

if [ "${visualize_bigwig_in_reports}" == "yes" ]; then
	mkdir -p publish
	mv ${name}.bw publish/.
fi
"""

}



process BAM_Analysis_Module_HISAT2_Genome_Browser {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "genome_browser_hisat2/$filename"}
input:
 file bigwigs from g252_142_publishBw10_g252_145.collect()
 file group_file from g_295_1_g252_145

output:
 file "*"  into g252_145_bigWig_file00

container 'quay.io/viascientific/python-basics:2.0'


script:
try {
    myVariable = group_file
} catch (MissingPropertyException e) {
    group_file = ""
}

genome_build_short= ""
if (params.genome_build == "mousetest_mm10"){
    genome_build_short= "mm10"
} else if (params.genome_build == "human_hg19_refseq"){
    genome_build_short = "hg19"
} else if (params.genome_build == "human_hg38_gencode_v28"){
    genome_build_short = "hg38"
} else if (params.genome_build == "human_hg38_gencode_v34"){
    genome_build_short = "hg38"
} else if (params.genome_build == "mouse_mm10_refseq"){
    genome_build_short = "mm10"
} else if (params.genome_build == "mouse_mm10_gencode_m25"){
    genome_build_short = "mm10"
} else if (params.genome_build == "rat_rn6_refseq"){
    genome_build_short = "rn6"
} else if (params.genome_build == "rat_rn6_ensembl_v86"){
    genome_build_short = "rn6"
} else if (params.genome_build == "zebrafish_GRCz11_ensembl_v95"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_refseq"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "zebrafish_GRCz11_v4.3.2"){
    genome_build_short = "GRCz11"
} else if (params.genome_build == "c_elegans_ce11_ensembl_ws245"){
    genome_build_short = "ce11"
} else if (params.genome_build == "d_melanogaster_dm6_refseq"){
    genome_build_short = "dm6"
} else if (params.genome_build == "s_cerevisiae_sacCer3_refseq"){
    genome_build_short = "sacCer3"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v31"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "s_pombe_ASM294v2_ensembl_v51"){
    genome_build_short = "ASM294v2"
} else if (params.genome_build == "e_coli_ASM584v2_refseq"){
    genome_build_short = "ASM584v2"
} else if (params.genome_build == "dog_canFam3_refseq"){
    genome_build_short = "canFam3"
} 

"""

#!/usr/bin/env python

import glob,json,os,sys,csv,random,shutil,requests,subprocess

def check_url_existence(url):
    try:
        response = requests.head(url, allow_redirects=True)
        # Check if the status code is in the success range (200-399)
        return 200 <= response.status_code < 400
    except requests.ConnectionError:
        return False

def get_lib_val(str, lib):
    for key, value in lib.items():
        if key.lower() in str.lower():
            return value
    return None

def find_and_move_folders_with_bw_files(start_dir):
    bigwig_dir = "bigwigs"
    if os.path.exists(bigwig_dir)!=True:
        os.makedirs(bigwig_dir)
    subprocess.getoutput("cp ${bigwigs} bigwigs/. ")

def Generate_HubFile(groupfile):
    # hub.txt
    Hub = open("hub.txt", "w")
    Hub.write("hub UCSCHub \\n")
    Hub.write("shortLabel UCSCHub \\n")
    Hub.write("longLabel UCSCHub \\n")
    Hub.write("genomesFile genomes.txt \\n")
    Hub.write("email support@viascientific.com \\n")
    Hub.write("\\n")
    Hub.close()

    # genomes.txt
    genomes = open("genomes.txt", "w")
    genomes.write("genome ${genome_build_short} \\n")
    genomes.write("trackDb bigwigs/trackDb.txt \\n")
    genomes.write("\\n")
    genomes.close()
    #trackDb = open("bigwigs/trackDb.txt", "w")
    path = r'bigwigs/*.bw'
    files = glob.glob(path)
    files.sort()
    sample={}
    for i in files:
        temp=i.split('.')[0]
        temp=temp.replace('bigwigs/','')
        if temp in sample.keys():
            sample[temp].append(i.replace('bigwigs/',''))
        else:
            sample[temp]=[]
            sample[temp].append(i.replace('bigwigs/',''))
    second_layer_indent=" "*2
    third_layer_indent = " " * 14
    if groupfile == "" or os.stat(groupfile).st_size == 0:
        #No GroupFile
        trackDb = open("bigwigs/trackDb.txt", "w")
        for i in sample.keys():
            trackDb.write('track %s\\n' %i)
            trackDb.write('shortLabel %s\\n' %i)
            trackDb.write('longLabel %s\\n' %i)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for j in sample[i]:
                trackDb.write(second_layer_indent+'track %s\\n' % j)
                trackDb.write(second_layer_indent+'bigDataUrl %s\\n' % j)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % j)
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %i)
                trackDb.write('\\n')
        trackDb.close()
    else:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_to_sample={}
        for cond in condition:
            if cond not in condition_to_sample.keys():
                condition_to_sample[cond]=[]
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
            else:
                for i in sample.keys():
                    if data_parsed[i]['group']==cond:
                        condition_to_sample[cond].append(i)
        trackDb = open("bigwigs/trackDb.txt", "w")
        for cond in condition:
            trackDb.write('track %s\\n' %cond)
            trackDb.write('shortLabel %s\\n' %cond)
            trackDb.write('longLabel %s\\n' %cond)
            trackDb.write('visibility full\\n')
            trackDb.write('compositeTrack on\\n')
            trackDb.write('type bigWig\\n')
            trackDb.write('\\n')
            for samp in condition_to_sample[cond]:
                trackDb.write(second_layer_indent+'track %s\\n' % samp)
                trackDb.write(second_layer_indent+'shortLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'longLabel %s\\n' % samp)
                trackDb.write(second_layer_indent+'view Signal \\n')
                trackDb.write(second_layer_indent+'visibility full \\n')
                trackDb.write(second_layer_indent+'viewLimits 0:20 \\n')
                trackDb.write(second_layer_indent+'autoScale on \\n')
                trackDb.write(second_layer_indent+'maxHeightPixels 128:20:8 \\n')
                trackDb.write(second_layer_indent+'configurable on \\n')
                trackDb.write(second_layer_indent+'type bigWig\\n')
                trackDb.write(second_layer_indent+'parent %s\\n' %cond)
                trackDb.write('\\n')
                for file in sample[samp]:
                    trackDb.write(third_layer_indent + 'track %s\\n' % file)
                    trackDb.write(third_layer_indent + 'bigDataUrl %s\\n' % file)
                    trackDb.write(third_layer_indent + 'shortLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'longLabel %s\\n' % file)
                    trackDb.write(third_layer_indent + 'type bigWig\\n')
                    trackDb.write(third_layer_indent+'parent %s\\n' %samp)

                    trackDb.write('\\n')

        trackDb.close()


def Generating_Json_files(groupfile):
    publishWebDir = '{{DNEXT_WEB_REPORT_DIR}}/' + 'genome_browser_hisat2' + "/bigwigs"
    locusLib = {}
    # MYC locations
    locusLib["hg19"] = "chr8:128,746,315-128,755,680"
    locusLib["hg38"] = "chr8:127,733,434-127,744,951"
    locusLib["mm10"] = "chr15:61,983,341-61,992,361"
    locusLib["rn6"] = "chr7:102,584,313-102,593,240"
    locusLib["dm6"] = "chrX:3,371,159-3,393,697"
    locusLib["canFam3"] = "chr13:25,198,772-25,207,309"

    cytobandLib = {}
    cytobandLib["hg19"] = "https://igv-genepattern-org.s3.amazonaws.com/genomes/seq/hg19/cytoBand.txt"
    cytobandLib["hg38"] = "https://s3.amazonaws.com/igv.org.genomes/hg38/annotations/cytoBandIdeo.txt.gz"
    cytobandLib["mm10"] = "https://s3.amazonaws.com/igv.broadinstitute.org/annotations/mm10/cytoBandIdeo.txt.gz"
    cytobandLib["rn6"] = "https://s3.amazonaws.com/igv.org.genomes/rn6/cytoBand.txt.gz"
    cytobandLib["dm6"] = "https://s3.amazonaws.com/igv.org.genomes/dm6/cytoBandIdeo.txt.gz"
    cytobandLib["ce11"] = "https://s3.amazonaws.com/igv.org.genomes/ce11/cytoBandIdeo.txt.gz"
    cytobandLib["canFam3"] = "https://s3.amazonaws.com/igv.org.genomes/canFam3/cytoBandIdeo.txt.gz"

    # Get the basename of the original path
    gtf_source_base_name = os.path.basename("${params.gtf_source}")
    gtf_source_sorted = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz")
    print(gtf_source_sorted)
    gtf_source_sorted_index = os.path.join(os.path.dirname("${params.gtf_source}"), f"{os.path.splitext(gtf_source_base_name)[0]}.sorted.gtf.gz.tbi")
    print(gtf_source_sorted_index)


    data = {}
    data["reference"] = {}
    data["reference"]["id"] = "${params.genome_build}"
    data["reference"]["name"] = "${params.genome_build}"
    data["reference"]["fastaURL"] = "${params.genome_source}"
    data["reference"]["indexURL"] = "${params.genome_source}.fai"
    cytobandurl = get_lib_val("${params.genome_build}", cytobandLib)
    locusStr = get_lib_val("${params.genome_build}", locusLib)
    if cytobandurl is not None:
        data["reference"]["cytobandURL"] = cytobandurl
    if locusStr is not None:
        data["locus"] = []
        data["locus"].append(locusStr)
    data["tracks"] = []
    # prepare gtf Track
    gtfTrack = {}
    gtfTrack["name"] = "${params.genome_build}"
    gtfTrack["gtf"] = "gtf"
    if check_url_existence(gtf_source_sorted):
        gtfTrack["url"] = gtf_source_sorted
    else:
        gtfTrack["url"] = "${params.gtf_source}"
    if check_url_existence(gtf_source_sorted_index):
        gtfTrack["indexURL"] = gtf_source_sorted_index

    # prepare cytobands Track
    if groupfile and os.path.isfile(groupfile) and os.stat(groupfile).st_size != 0:
        samplesheet = open(groupfile)
        table = csv.reader(samplesheet, delimiter="\t", quotechar='\\"')
        table_parsed = []
        data_parsed = dict()
        for rows in table:
            table_parsed.append(rows)
        data_column = table_parsed[0]
        table_parsed.pop(0)
        for rows in table_parsed:
            data_parsed[rows[0]] = dict()
            for i in range(len(data_column)):
                data_parsed[rows[0]][data_column[i]] = rows[i]
        condition = list()
        samplesheet.close()
        for j in data_parsed.keys():
            if data_parsed[j]['group'] not in condition:
                condition.append(data_parsed[j]['group'])
        condition_color_dict = dict()
        for cond in condition:
            r = lambda: random.randint(0, 255)
            condition_color_dict[cond] = '#%02X%02X%02X' % (r(), r(), r())

    # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                if basename in data_parsed:
                    newdata[file] = data_parsed[basename]
                    newdata[file]["fullname"] = file

        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": condition_color_dict[newdata[j]['group']],
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()
    else:

        # get all the files in directories
        newdata = {}
        for filepath in glob.iglob('./bigwigs/*.bw', recursive=True):
            if os.path.isfile(filepath):
                file = os.path.basename(filepath)
                basename = file.split('.')[0]
                newdata[file] = basename
                newdata[file] = file
        tracks = []
        for j in newdata.keys():
            parts = j.split(".")
            sortName = parts[0]
            sortGroup = parts[-2] + "_" + parts[-1]
            r = lambda: random.randint(0, 255)
            tracktoadd = {
                "url": publishWebDir + "/" + j,
                "color": '#%02X%02X%02X' % (r(), r(), r()),
                "format": "bigwig",
                "filename": j,
                "sortName": sortName,
                "sortGroup": sortGroup,
                "sourceType": "file",
                "height": 50,
                "autoscale": True,
                "order": 4}
            tracks.append(tracktoadd)

        tracks.sort(key=lambda x: x["sortName"])
        tracks.sort(key=lambda x: x["sortGroup"])

        data['tracks'] = tracks
        data['tracks'].insert(0, gtfTrack)

        with open('./access_IGV.json', 'w') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        f.close()

if __name__ == "__main__":
    find_and_move_folders_with_bw_files(".")
    Generate_HubFile(groupfile="${group_file}")
    Generating_Json_files(groupfile="${group_file}")

"""
}

//* params.bed =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 10
}
//* platform
//* platform
//* autofill

process BAM_Analysis_Module_HISAT2_RSeQC {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "rseqc_hisat2/$filename"}
input:
 set val(name), file(bam), file(bai) from g252_143_bam_bai00_g252_134
 file bed from g245_54_bed31_g252_134
 val mate from g_347_mate12_g252_134

output:
 file "*"  into g252_134_outputFileOut09_g_177

container 'quay.io/viascientific/rseqc:1.0'

when:
(params.run_RSeQC && (params.run_RSeQC == "yes")) || !params.run_RSeQC

script:
run_bam_stat = params.BAM_Analysis_Module_HISAT2_RSeQC.run_bam_stat
run_read_distribution = params.BAM_Analysis_Module_HISAT2_RSeQC.run_read_distribution
run_inner_distance = params.BAM_Analysis_Module_HISAT2_RSeQC.run_inner_distance
run_junction_annotation = params.BAM_Analysis_Module_HISAT2_RSeQC.run_junction_annotation
run_junction_saturation = params.BAM_Analysis_Module_HISAT2_RSeQC.run_junction_saturation
//run_geneBody_coverage and run_infer_experiment needs subsampling
run_geneBody_coverage = params.BAM_Analysis_Module_HISAT2_RSeQC.run_geneBody_coverage
run_infer_experiment = params.BAM_Analysis_Module_HISAT2_RSeQC.run_infer_experiment
"""
if [ "$run_bam_stat" == "true" ]; then bam_stat.py  -i ${bam} > ${name}.bam_stat.txt; fi
if [ "$run_read_distribution" == "true" ]; then read_distribution.py  -i ${bam} -r ${bed}> ${name}.read_distribution.out; fi


if [ "$run_infer_experiment" == "true" -o "$run_geneBody_coverage" == "true" ]; then
	numAlignedReads=\$(samtools view -c -F 4 $bam)

	if [ "\$numAlignedReads" -gt 1000000 ]; then
    	echo "Read number is greater than 1000000. Subsampling..."
    	finalRead=1000000
    	fraction=\$(samtools idxstats  $bam | cut -f3 | awk -v ct=\$finalRead 'BEGIN {total=0} {total += \$1} END {print ct/total}')
    	samtools view -b -s \${fraction} $bam > ${name}_sampled.bam
    	samtools index ${name}_sampled.bam
    	if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i ${name}_sampled.bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i ${name}_sampled.bam  -r $bed -o ${name}; fi
	else
		if [ "$run_infer_experiment" == "true" ]; then infer_experiment.py -i $bam  -r $bed > ${name}; fi
		if [ "$run_geneBody_coverage" == "true" ]; then geneBody_coverage.py -i $bam  -r $bed -o ${name}; fi
	fi

fi


if [ "${mate}" == "pair" ]; then
	if [ "$run_inner_distance" == "true" ]; then inner_distance.py -i $bam  -r $bed -o ${name}.inner_distance > stdout.txt; fi
	if [ "$run_inner_distance" == "true" ]; then head -n 2 stdout.txt > ${name}.inner_distance_mean.txt; fi
fi
if [ "$run_junction_annotation" == "true" ]; then junction_annotation.py -i $bam  -r $bed -o ${name}.junction_annotation 2> ${name}.junction_annotation.log; fi
if [ "$run_junction_saturation" == "true" ]; then junction_saturation.py -i $bam  -r $bed -o ${name}; fi
if [ -e class.log ] ; then mv class.log ${name}_class.log; fi
if [ -e log.txt ] ; then mv log.txt ${name}_log.txt; fi
if [ -e stdout.txt ] ; then mv stdout.txt ${name}_stdout.txt; fi


"""

}

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 12
}
//* platform
//* platform
//* autofill

process MultiQC {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /multiqc_report.html$/) "multiqc/$filename"}
input:
 file "rsem/*" from g250_26_rsemOut01_g_177.flatten().toList()
 file "star/*" from g264_31_logFinalOut62_g_177.flatten().toList()
 file "fastqc/*" from g257_28_FastQCout04_g_177.flatten().toList()
 file "rseqc_rsem/*" from g251_134_outputFileOut07_g_177.flatten().toList()
 file "rseqc_star/*" from g253_134_outputFileOut08_g_177.flatten().toList()
 file "rseqc_hisat/*" from g252_134_outputFileOut09_g_177.flatten().toList()
 file "kallisto/*" from g248_36_outputDir013_g_177.flatten().toList()
 file "rseqc_kallisto/*" from g255_134_outputFileOut014_g_177.flatten().toList()
 file "after_adapter_removal/*" from g257_31_FastQCout015_g_177.flatten().toList()
 file "salmon/*" from g268_44_outputDir016_g_177.flatten().toList()
 file "salmon_bam_count/*" from g276_9_outputDir017_g_177.flatten().toList()

output:
 file "multiqc_report.html" optional true  into g_177_outputHTML00


container 'quay.io/biocontainers/multiqc:1.21--pyhdfd78af_0'
script:
multiqc_parameters = params.MultiQC.multiqc_parameters
"""
multiqc ${multiqc_parameters}  -d -dd 2 .
"""

}

//* params.pdfbox_path =  ""  //* @input
//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 32
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 240
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "short"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 500
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
}
//* platform
//* autofill

process BAM_Analysis_Module_HISAT2_Picard {

input:
 set val(name), file(bam) from g249_13_sorted_bam20_g252_121

output:
 file "*_metrics"  into g252_121_outputFileOut00_g252_82
 file "results/*.pdf"  into g252_121_outputFilePdf12_g252_82

container 'quay.io/viascientific/picard:1.0'

when:
(params.run_Picard_CollectMultipleMetrics && (params.run_Picard_CollectMultipleMetrics == "yes")) || !params.run_Picard_CollectMultipleMetrics

script:
"""
picard CollectMultipleMetrics OUTPUT=${name}_multiple.out VALIDATION_STRINGENCY=LENIENT INPUT=${bam}
mkdir results && java -jar ${params.pdfbox_path} PDFMerger *.pdf results/${name}_multi_metrics.pdf
"""
}


process BAM_Analysis_Module_HISAT2_Picard_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "picard_summary_hisat2/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tsv$/) "rseqc_summary_hisat2/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /results\/.*.pdf$/) "picard_summary_pdf_hisat2/$filename"}
input:
 file picardOut from g252_121_outputFileOut00_g252_82.collect()
 val mate from g_347_mate11_g252_82
 file picardPdf from g252_121_outputFilePdf12_g252_82.collect()

output:
 file "*.tsv"  into g252_82_outputFileTSV00
 file "results/*.pdf"  into g252_82_outputFilePdf11

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage; 
use Data::Dumper;

runCommand("mkdir results && mv *.pdf results/. ");

my $indir = $ENV{'PWD'};
my $outd = $ENV{'PWD'};
my @files = ();
my @outtypes = ("CollectRnaSeqMetrics", "alignment_summary_metrics", "base_distribution_by_cycle_metrics", "insert_size_metrics", "quality_by_cycle_metrics", "quality_distribution_metrics" );

foreach my $outtype (@outtypes)
{
my $ext="_multiple.out";
$ext.=".$outtype" if ($outtype ne "CollectRnaSeqMetrics");
@files = <$indir/*$ext>;

my @rowheaders=();
my @libs=();
my %metricvals=();
my %histvals=();

my $pdffile="";
my $libname="";
foreach my $d (@files){
  my $libname=basename($d, $ext);
  print $libname."\\n";
  push(@libs, $libname); 
  getMetricVals($d, $libname, \\%metricvals, \\%histvals, \\@rowheaders);
}

my $sizemetrics = keys %metricvals;
write_results("$outd/$outtype.stats.tsv", \\@libs,\\%metricvals, \\@rowheaders, "metric") if ($sizemetrics>0);
my $sizehist = keys %histvals;
write_results("$outd/$outtype.hist.tsv", \\@libs,\\%histvals, "none", "nt") if ($sizehist>0);

}

sub write_results
{
  my ($outfile, $libs, $vals, $rowheaders, $name )=@_;
  open(OUT, ">$outfile");
  print OUT "$name\\t".join("\\t", @{$libs})."\\n";
  my $size=0;
  $size=scalar(@{${$vals}{${$libs}[0]}}) if(exists ${$libs}[0] and exists ${$vals}{${$libs}[0]} );
  
  for (my $i=0; $i<$size;$i++)
  { 
    my $rowname=$i;
    $rowname = ${$rowheaders}[$i] if ($name=~/metric/);
    print OUT $rowname;
    foreach my $lib (@{$libs})
    {
      print OUT "\\t".${${$vals}{$lib}}[$i];
    } 
    print OUT "\\n";
  }
  close(OUT);
}

sub getMetricVals{
  my ($filename, $libname, $metricvals, $histvals,$rowheaders)=@_;
  if (-e $filename){
     my $nextisheader=0;
     my $nextisvals=0;
     my $nexthist=0;
     open(IN, $filename);
     while(my $line=<IN>)
     {
       chomp($line);
       @{$rowheaders}=split(/\\t/, $line) if ($nextisheader && !scalar(@{$rowheaders})); 
       if ($nextisvals) {
         @{${$metricvals}{$libname}}=split(/\\t/, $line);
         $nextisvals=0;
       }
       if($nexthist){
          my @vals=split(/[\\s\\t]+/,$line); 
          push(@{${$histvals}{$libname}}, $vals[1]) if (exists $vals[1]);
       }
       $nextisvals=1 if ($nextisheader); $nextisheader=0;
       $nextisheader=1 if ($line=~/METRICS CLASS/);
       $nexthist=1 if ($line=~/normalized_position/);
     } 
  }
  
}


sub runCommand {
	my ($com) = @_;
	if ($com eq ""){
		return "";
    }
    my $error = system(@_);
	if   ($error) { die "Command failed: $error $com\\n"; }
    else          { print "Command successful: $com\\n"; }
}
'''

}

igv_extention_factor = params.BAM_Analysis_Module_HISAT2_IGV_BAM2TDF_converter.igv_extention_factor
igv_window_size = params.BAM_Analysis_Module_HISAT2_IGV_BAM2TDF_converter.igv_window_size

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 24
}
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} else if ($HOSTNAME == "hpc.umassmed.edu"){
    $TIME = 400
    $CPU  = 1
    $MEMORY = 32
    $QUEUE = "long"
} 
//* platform
//* autofill

process BAM_Analysis_Module_HISAT2_IGV_BAM2TDF_converter {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*.tdf$/) "igvtools_hisat2/$filename"}
input:
 val mate from g_347_mate10_g252_131
 set val(name), file(bam) from g249_13_sorted_bam21_g252_131
 file genomeSizes from g245_54_genomeSizes22_g252_131

output:
 file "*.tdf"  into g252_131_outputFileOut00

when:
(params.run_IGV_TDF_Conversion && (params.run_IGV_TDF_Conversion == "yes")) || !params.run_IGV_TDF_Conversion

script:
pairedText = (params.nucleicAcidType == "dna" && mate == "pair") ? " --pairs " : ""
nameAll = bam.toString()
if (nameAll.contains('_sorted.bam')) {
    runSamtools = "samtools index ${nameAll}"
    nameFinal = nameAll
} else {
    runSamtools = "samtools sort -o ${name}_sorted.bam $bam && samtools index ${name}_sorted.bam "
    nameFinal = "${name}_sorted.bam"
}
"""
$runSamtools
igvtools count -w ${igv_window_size} -e ${igv_extention_factor} ${pairedText} ${nameFinal} ${name}.tdf ${genomeSizes}
"""
}

//* params.gtf =  ""  //* @input


process Bam_Quantify_Module_HISAT2_featureCounts {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*$/) "featureCounts_after_hisat2/$filename"}
input:
 set val(name), file(bam) from g249_13_sorted_bam20_g280_1
 val paired from g_347_mate11_g280_1
 each run_params from g280_0_run_parameters02_g280_1
 file gtf from g245_54_gtfFile03_g280_1

output:
 file "*"  into g280_1_outputFileTSV00_g280_2

script:
pairText = ""
if (paired == "pair"){
    pairText = "-p"
}

run_name = run_params["run_name"] 
run_parameters = run_params["run_parameters"] 

"""
featureCounts ${pairText} ${run_parameters} -a ${gtf} -o ${name}@${run_name}@fCounts.txt ${bam}
## remove first line
sed -i '1d' ${name}@${run_name}@fCounts.txt

"""
}

//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 30
    $CPU  = 1
    $MEMORY = 10
    $QUEUE = "short"
}
//* platform
//* autofill

process Bam_Quantify_Module_HISAT2_featureCounts_summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_featureCounts.tsv$/) "featureCounts_after_hisat2_summary/$filename"}
publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /.*_featureCounts.sum.tsv$/) "featureCounts_after_hisat2_details/$filename"}
input:
 file featureCountsOut from g280_1_outputFileTSV00_g280_2.collect()

output:
 file "*_featureCounts.tsv"  into g280_2_outputFile00_g306_25, g280_2_outputFile00_g306_24
 file "*_featureCounts.sum.tsv"  into g280_2_outFileTSV11

shell:
'''
#!/usr/bin/env perl

# Step 1: Merge count files
my %tf = ( expected_count => 6 );
my @run_name=();
chomp(my $contents = `ls *@fCounts.txt`);
my @files = split(/[\\n]+/, $contents);
foreach my $file (@files){
    $file=~/(.*)\\@(.*)\\@fCounts\\.txt/;
    my $runname = $2;
    push(@run_name, $runname) unless grep{$_ eq $runname} @run_name;
}


my @expectedCount_ar = ("expected_count");
for($l = 0; $l <= $#run_name; $l++) {
    my $runName = $run_name[$l];
    for($ll = 0; $ll <= $#expectedCount_ar; $ll++) {
        my $expectedCount = $expectedCount_ar[$ll];
    
        my @a=();
        my %b=();
        my %c=();
        my $i=0;
        chomp(my $contents = `ls *\\@${runName}\\@fCounts.txt`);
        my @files = split(/[\\n]+/, $contents);
        foreach my $file (@files){
        $i++;
        $file=~/(.*)\\@${runName}\\@fCounts\\.txt/;
        my $libname = $1; 
        $a[$i]=$libname;
        open IN, $file;
            $_=<IN>;
            while(<IN>){
                my @v=split; 
                $b{$v[0]}{$i}=$v[$tf{$expectedCount}];
                $c{$v[0]}=$v[5]; #length column
            }
            close IN;
        }
        my $outfile="$runName"."_featureCounts.tsv";
        open OUT, ">$outfile";
        if ($runName eq "transcript_id") {
            print OUT "transcript\tlength";
        } else {
            print OUT "gene\tlength";
        }
    
        for(my $j=1;$j<=$i;$j++) {
            print OUT "\t$a[$j]";
        }
        print OUT "\n";
    
        foreach my $key (keys %b) {
            print OUT "$key\t$c{$key}";
            for(my $j=1;$j<=$i;$j++){
                print OUT "\t$b{$key}{$j}";
            }
            print OUT "\n";
        }
        close OUT;
         
    }
}


	

# Step 2: Merge summary files
for($l = 0; $l <= $#run_name; $l++) {
    my $runName = $run_name[$l];
    my @a=();
    my %b=();
    my $i=0;
    chomp(my $contents = `ls *\\@${runName}\\@fCounts.txt.summary`);
    my @files = split(/[\\n]+/, $contents);
    foreach my $file (@files){
        $i++;
        $file=~/(.*)\\@${runName}\\@fCounts\\.txt\\.summary/;
        my $libname = $1; 
        $a[$i]=$libname;
        open IN, $file;
        $_=<IN>;
        while(<IN>){
            my @v=split; 
            $b{$v[0]}{$i}=$v[1];
        }
        close IN;
    }
    my $outfile="$runName"."_featureCounts.sum.tsv";
    open OUT, ">$outfile";
    print OUT "criteria";
    for(my $j=1;$j<=$i;$j++) {
        print OUT "\t$a[$j]";
    }
    print OUT "\n";
    
    foreach my $key (keys %b) {
        print OUT "$key";
        for(my $j=1;$j<=$i;$j++){
            print OUT "\t$b{$key}{$j}";
        }
        print OUT "\n";
    }
    close OUT;
}

'''
}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_HISAT2_featurecounts_Prepare_DESeq2 {

input:
 file counts from g280_2_outputFile00_g306_24
 file groups_file from g_295_1_g306_24
 file compare_file from g_294_2_g306_24
 val run_DESeq2 from g_307_3_g306_24

output:
 file "DE_reports"  into g306_24_outputFile00_g306_37
 val "_des"  into g306_24_postfix10_g306_33
 file "DE_reports/outputs/*_all_deseq2_results.tsv"  into g306_24_outputFile21_g306_33

container 'quay.io/viascientific/de_module:4.0'

when:
run_DESeq2 == 'yes'

script:

feature_type = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_distribution
include_all2all = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_all2all
include_pca = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_pca

filter_type = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.filter_type
min_count = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.min_count
min_samples = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.min_samples
min_counts_per_sample = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.min_counts_per_sample
excluded_events = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.excluded_events

include_batch_correction = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_batch_correction
batch_correction_column = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.batch_correction_column
batch_correction_group_column = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.batch_normalization_algorithm

transformation = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.transformation
pca_color = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.pca_color
pca_shape = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.pca_shape
pca_fill = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.pca_fill
pca_transparency = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.pca_transparency
pca_label = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.pca_label

include_deseq2 = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_deseq2
input_mode = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.input_mode
design = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.design
fitType = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.fitType
use_batch_corrected_in_DE = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.use_batch_corrected_in_DE
apply_shrinkage = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.apply_shrinkage
shrinkage_type = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.shrinkage_type
include_volcano = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_volcano
include_ma = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_ma
include_heatmap = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_heatmap

padj_significance_cutoff = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.fc_significance_cutoff
padj_floor = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.padj_floor
fc_ceiling = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.fc_ceiling

convert_names = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.convert_names
count_file_names = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.count_file_names
converted_name = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.converted_name
org_db = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.org_db
num_labeled = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.num_labeled
highlighted_genes = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.highlighted_genes
include_volcano_highlighted = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_volcano_highlighted
include_ma_highlighted = params.DE_module_HISAT2_featurecounts_Prepare_DESeq2.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name, org_db},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm, use_batch_corrected_in_DE},{include_batch_correction="false"},{include_deseq2="true", design, fitType, apply_shrinkage, shrinkage_type, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, org_db, num_labeled, highlighted_genes},{include_deseq2="false"},{apply_shrinkage="true", shrinkage_type},{apply_shrinkage="false"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples, min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{design, fitType, use_batch_corrected_in_DE, apply_shrinkage, shrinkage_type},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name, org_db},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_deseq2 = include_deseq2 == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
apply_shrinkage = apply_shrinkage == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir reports
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_DESeq2.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-DESeq2 ${include_deseq2} --input-mode ${input_mode} --design '${design}' --fitType ${fitType} --use-batch-correction-in-DE ${use_batch_corrected_in_DE} --apply-shrinkage ${apply_shrinkage} --shrinkage-type ${shrinkage_type} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --org-db ${org_db} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}


//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_HISAT2_featurecounts_Prepare_LimmaVoom {

input:
 file counts from g280_2_outputFile00_g306_25
 file groups_file from g_295_1_g306_25
 file compare_file from g_294_2_g306_25
 val run_limmaVoom from g_357_3_g306_25

output:
 file "DE_reports"  into g306_25_outputFile00_g306_39
 val "_lv"  into g306_25_postfix10_g306_41
 file "DE_reports/outputs/*_all_limmaVoom_results.tsv"  into g306_25_outputFile21_g306_41

container 'quay.io/viascientific/de_module:4.0'

when:
run_limmaVoom == 'yes'

script:

feature_type = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.feature_type

countsFile = ""
listOfFiles = counts.toString().split(" ")
if (listOfFiles.size() > 1){
	countsFile = listOfFiles[0]
	for(item in listOfFiles){
    	if (feature_type.equals('gene') && item.startsWith("gene") && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;
    	} else if (feature_type.equals('transcript') && (item.startsWith("isoforms") || item.startsWith("transcript")) && item.toLowerCase().indexOf("tpm") < 0) {
    		countsFile = item
    		break;	
    	}
	}
} else {
	countsFile = counts
}

include_distribution = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_distribution
include_all2all = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_all2all
include_pca = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_pca

filter_type = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.filter_type
min_count = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.min_count
min_samples = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.min_samples
min_counts_per_sample = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.min_counts_per_sample
excluded_events = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.excluded_events

include_batch_correction = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_batch_correction
batch_correction_column = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.batch_correction_column
batch_correction_group_column = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.batch_correction_group_column
batch_normalization_algorithm = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.batch_normalization_algorithm

transformation = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.transformation
pca_color = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.pca_color
pca_shape = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.pca_shape
pca_fill = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.pca_fill
pca_transparency = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.pca_transparency
pca_label = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.pca_label

include_limma = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_limma
use_batch_corrected_in_DE = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.use_batch_corrected_in_DE
normalization_method = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.normalization_method
logratioTrim = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.logratioTrim
sumTrim = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.sumTrim
Acutoff = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.Acutoff
doWeighting = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.doWeighting
p = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.p
include_volcano = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_volcano
include_ma = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_ma
include_heatmap = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_heatmap

padj_significance_cutoff = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.padj_significance_cutoff
fc_significance_cutoff = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.fc_significance_cutoff
padj_floor = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.padj_floor
fc_ceiling = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.fc_ceiling

convert_names = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.convert_names
count_file_names = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.count_file_names
converted_name = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.converted_name
num_labeled = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.num_labeled
highlighted_genes = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.highlighted_genes
include_volcano_highlighted = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_volcano_highlighted
include_ma_highlighted = params.DE_module_HISAT2_featurecounts_Prepare_LimmaVoom.include_ma_highlighted

//* @style @condition:{convert_names="true", count_file_names, converted_name},{convert_names="false"},{include_batch_correction="true", batch_correction_column, batch_correction_group_column, batch_normalization_algorithm,use_batch_corrected_in_DE},{include_batch_correction="false"},{include_limma="true", normalization_method, logratioTrim, sumTrim, doWeighting, Acutoff, include_volcano, include_ma, include_heatmap, padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling, convert_names, count_file_names, converted_name, num_labeled, highlighted_genes},{include_limma="false"},{normalization_method="TMM", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="TMMwsp", logratioTrim, sumTrim, doWeighting, Acutoff},{normalization_method="RLE"},{normalization_method="upperquartile", p},{normalization_method="none"},{include_pca="true", transformation, pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_pca="false"} @multicolumn:{include_distribution, include_all2all, include_pca},{filter_type, min_count, min_samples,min_counts_per_sample},{include_batch_correction, batch_correction_column, batch_correction_group_column, batch_normalization_algorithm},{pca_color, pca_shape, pca_fill, pca_transparency, pca_label},{include_limma, use_batch_corrected_in_DE},{normalization_method,logratioTrim,sumTrim,doWeighting,Acutoff,p},{padj_significance_cutoff, fc_significance_cutoff, padj_floor, fc_ceiling},{convert_names, count_file_names, converted_name},{include_volcano, include_ma, include_heatmap}{include_volcano_highlighted,include_ma_highlighted} 

include_distribution = include_distribution == 'true' ? 'TRUE' : 'FALSE'
include_all2all = include_all2all == 'true' ? 'TRUE' : 'FALSE'
include_pca = include_pca == 'true' ? 'TRUE' : 'FALSE'
include_batch_correction = include_batch_correction == 'true' ? 'TRUE' : 'FALSE'
include_limma = include_limma == 'true' ? 'TRUE' : 'FALSE'
use_batch_corrected_in_DE = use_batch_corrected_in_DE  == 'true' ? 'TRUE' : 'FALSE'
convert_names = convert_names == 'true' ? 'TRUE' : 'FALSE'
include_ma = include_ma == 'true' ? 'TRUE' : 'FALSE'
include_volcano = include_volcano == 'true' ? 'TRUE' : 'FALSE'
include_heatmap = include_heatmap == 'true' ? 'TRUE' : 'FALSE'

doWeighting = doWeighting == 'true' ? 'TRUE' : 'FALSE'
TMM_args = normalization_method.equals('TMM') || normalization_method.equals('TMMwsp') ? '--logratio-trim ' + logratioTrim + ' --sum-trim ' + sumTrim + ' --do-weighting ' + doWeighting + ' --A-cutoff="' + Acutoff + '"' : ''
upperquartile_args = normalization_method.equals('upperquartile') ? '--p ' + p : ''

excluded_events = excluded_events.replace("\n", " ").replace(',', ' ')

pca_color_arg = pca_color.equals('') ? '' : '--pca-color ' + pca_color
pca_shape_arg = pca_shape.equals('') ? '' : '--pca-shape ' + pca_shape
pca_fill_arg = pca_fill.equals('') ? '' : '--pca-fill ' + pca_fill
pca_transparency_arg = pca_transparency.equals('') ? '' : '--pca-transparency ' + pca_transparency
pca_label_arg = pca_label.equals('') ? '' : '--pca-label ' + pca_label

highlighted_genes = highlighted_genes.replace("\n", " ").replace(',', ' ')

excluded_events_arg = excluded_events.equals('') ? '' : '--excluded-events ' + excluded_events
highlighted_genes_arg = highlighted_genes.equals('') ? '' : '--highlighted-genes ' + highlighted_genes
include_ma_highlighted = include_ma_highlighted == 'true' ? 'TRUE' : 'FALSE'
include_volcano_highlighted = include_volcano_highlighted == 'true' ? 'TRUE' : 'FALSE'

threads = task.cpus

"""
mkdir inputs
mkdir outputs
cp ${groups_file} inputs/${groups_file}
cp ${compare_file} inputs/${compare_file}
cp ${countsFile} inputs/${countsFile}
cp inputs/${groups_file} inputs/.de_metadata.txt
cp inputs/${compare_file} inputs/.comparisons.tsv

prepare_limmaVoom.py \
--counts inputs/${countsFile} --groups inputs/${groups_file} --comparisons inputs/${compare_file} --feature-type ${feature_type} \
--include-distribution ${include_distribution} --include-all2all ${include_all2all} --include-pca ${include_pca} \
--filter-type ${filter_type} --min-counts-per-event ${min_count} --min-samples-per-event ${min_samples} --min-counts-per-sample ${min_counts_per_sample} \
--transformation ${transformation} ${pca_color_arg} ${pca_shape_arg} ${pca_fill_arg} ${pca_transparency_arg} ${pca_label_arg} \
--include-batch-correction ${include_batch_correction} --batch-correction-column ${batch_correction_column} --batch-correction-group-column ${batch_correction_group_column} --batch-normalization-algorithm ${batch_normalization_algorithm} \
--include-limma ${include_limma} \
--use-batch-correction-in-DE ${use_batch_corrected_in_DE} --normalization-method ${normalization_method} ${TMM_args} ${upperquartile_args} \
--include-volcano ${include_volcano} --include-ma ${include_ma} --include-heatmap ${include_heatmap} \
--padj-significance-cutoff ${padj_significance_cutoff} --fc-significance-cutoff ${fc_significance_cutoff} --padj-floor ${padj_floor} --fc-ceiling ${fc_ceiling} \
--convert-names ${convert_names} --count-file-names ${count_file_names} --converted-names ${converted_name} --num-labeled ${num_labeled} \
${highlighted_genes_arg} --include-volcano-highlighted ${include_volcano_highlighted} --include-ma-highlighted ${include_ma_highlighted} \
${excluded_events_arg} \
--threads ${threads}

mkdir DE_reports
mv *.Rmd DE_reports
mv *.html DE_reports
mv inputs DE_reports/
mv outputs DE_reports/
"""

}


process HISAT2_Module_HISAT2_Summary {

input:
 set val(name), file(alignSum) from g249_14_outputFileTxt10_g249_2.groupTuple()

output:
 file "*.tsv"  into g249_2_outputFile00_g249_10
 val "hisat2_alignment_sum"  into g249_2_name11_g249_10

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage; 
use Data::Dumper;

my %tsv;
my @headers = ();
my $name = "!{name}";


alteredAligned();

my @keys = keys %tsv;
my $summary = "$name"."_hisat_sum.tsv";
my $header_string = join("\\t", @headers);
`echo "$header_string" > $summary`;
foreach my $key (@keys){
	my $values = join("\\t", @{ $tsv{$key} });
	`echo "$values" >> $summary`;
}


sub alteredAligned
{
	my @files = qw(!{alignSum});
	my $multimappedSum;
	my $alignedSum;
	my $inputCountSum;
	push(@headers, "Sample");
    push(@headers, "Total Reads");
	push(@headers, "Multimapped Reads Aligned (HISAT2)");
	push(@headers, "Unique Reads Aligned (HISAT2)");
	foreach my $file (@files){
		my $multimapped;
		my $aligned;
		my $inputCount;
		chomp($inputCount = `cat $file | grep 'reads; of these:' | awk '{sum+=\\$1} END {print sum}'`);
		chomp($aligned = `cat $file | grep 'aligned.*exactly 1 time' | awk '{sum+=\\$1} END {print sum}'`);
		chomp($multimapped = `cat $file | grep 'aligned.*>1 times' | awk '{sum+=\\$1} END {print sum}'`);
		$multimappedSum += int($multimapped);
        $alignedSum += int($aligned);
        $inputCountSum += int($inputCount);
	}
	$tsv{$name} = [$name, $inputCountSum];
	push(@{$tsv{$name}}, $multimappedSum);
	push(@{$tsv{$name}}, $alignedSum);
}
'''

}


process HISAT2_Module_Merge_TSV_Files {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /${name}.tsv$/) "hisat2_summary/$filename"}
input:
 file tsv from g249_2_outputFile00_g249_10.collect()
 val outputFileName from g249_2_name11_g249_10.collect()

output:
 file "${name}.tsv"  into g249_10_outputFileTSV02_g_198

errorStrategy 'retry'
maxRetries 3

script:
name = outputFileName[0]
"""    
awk 'FNR==1 && NR!=1 {  getline; } 1 {print} ' *.tsv > ${name}.tsv
"""
}


process Adapter_Trimmer_Quality_Module_Quality_Filtering_Summary {

input:
 file logfile from g257_20_log_file10_g257_16.collect()
 val mate from g_347_mate11_g257_16

output:
 file "quality_filter_summary.tsv"  into g257_16_outputFileTSV07_g_198

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;

my @header;
my %all_files;
my %tsv;
my %headerHash;
my %headerText;

my $i = 0;
chomp( my $contents = `ls *.log` );
my @files = split( /[\\n]+/, $contents );
foreach my $file (@files) {
    $i++;
    my $mapper   = "";
    my $mapOrder = "1";
    if ($file =~ /(.*)\\.fastx_quality\\.log/){
        $mapper   = "fastx";
        $file =~ /(.*)\\.fastx_quality\\.log/;
        my $name = $1;    ##sample name
        push( @header, $mapper );
        my $in;
        my $out;
        chomp( $in =`cat $file | grep 'Input:' | awk '{sum+=\\$2} END {print sum}'` );
        chomp( $out =`cat $file | grep 'Output:' | awk '{sum+=\\$2} END {print sum}'` );
        $tsv{$name}{$mapper} = [ $in, $out ];
        $headerHash{$mapOrder} = $mapper;
        $headerText{$mapOrder} = [ "Total Reads", "Reads After Quality Filtering" ];
    } elsif ($file =~ /(.*)\\.trimmomatic_quality\\.log/){
        $mapper   = "trimmomatic";
        $file =~ /(.*)\\.trimmomatic_quality\\.log/;
        my $name = $1;    ##sample name
        push( @header, $mapper );
        my $in;
        my $out;
        if ( "!{mate}" eq "pair"){
            chomp( $in =`cat $file | grep 'Input Read Pairs:' | awk '{sum+=\\$4} END {print sum}'` );
            chomp( $out =`cat $file | grep 'Input Read Pairs:' | awk '{sum+=\\$7} END {print sum}'` );
        } else {
            chomp( $in =`cat $file | grep 'Input Reads:' | awk '{sum+=\\$3} END {print sum}'` );
            chomp( $out =`cat $file | grep 'Input Reads:' | awk '{sum+=\\$5} END {print sum}'` );
        }
        $tsv{$name}{$mapper} = [ $in, $out ];
        $headerHash{$mapOrder} = $mapper;
        $headerText{$mapOrder} = [ "Total Reads", "Reads After Quality Filtering" ];
    }
    
}

my @mapOrderArray = ( keys %headerHash );
my @sortedOrderArray = sort { $a <=> $b } @mapOrderArray;

my $summary          = "quality_filter_summary.tsv";
writeFile( $summary,          \\%headerText,       \\%tsv );

sub writeFile {
    my $summary    = $_[0];
    my %headerText = %{ $_[1] };
    my %tsv        = %{ $_[2] };
    open( OUT, ">$summary" );
    print OUT "Sample\\t";
    my @headArr = ();
    for my $mapOrder (@sortedOrderArray) {
        push( @headArr, @{ $headerText{$mapOrder} } );
    }
    my $headArrAll = join( "\\t", @headArr );
    print OUT "$headArrAll\\n";

    foreach my $name ( keys %tsv ) {
        my @rowArr = ();
        for my $mapOrder (@sortedOrderArray) {
            push( @rowArr, @{ $tsv{$name}{ $headerHash{$mapOrder} } } );
        }
        my $rowArrAll = join( "\\t", @rowArr );
        print OUT "$name\\t$rowArrAll\\n";
    }
    close(OUT);
}

'''
}


process Adapter_Trimmer_Quality_Module_Trimmer_Removal_Summary {

input:
 file logfile from g257_19_log_file10_g257_21.collect()
 val mate from g_347_mate11_g257_21

output:
 file "trimmer_summary.tsv"  into g257_21_outputFileTSV06_g_198

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;

my @header;
my %all_files;
my %tsv;
my %tsvDetail;
my %headerHash;
my %headerText;
my %headerTextDetail;

my $i = 0;
chomp( my $contents = `ls *.log` );

my @files = split( /[\\n]+/, $contents );
foreach my $file (@files) {
    $i++;
    my $mapOrder = "1";
    if ($file =~ /(.*)\\.fastx_trimmer\\.log/){
        $file =~ /(.*)\\.fastx_trimmer\\.log/;
        my $mapper   = "fastx_trimmer";
        my $name = $1;    ##sample name
        push( @header, $mapper );
        my $in;
        my $out;
        chomp( $in =`cat $file | grep 'Input:' | awk '{sum+=\\$2} END {print sum}'` );
        chomp( $out =`cat $file | grep 'Output:' | awk '{sum+=\\$2} END {print sum}'` );

        $tsv{$name}{$mapper} = [ $in, $out ];
        $headerHash{$mapOrder} = $mapper;
        $headerText{$mapOrder} = [ "Total Reads", "Reads After Trimmer" ];
    }
}

my @mapOrderArray = ( keys %headerHash );
my @sortedOrderArray = sort { $a <=> $b } @mapOrderArray;

my $summary          = "trimmer_summary.tsv";
writeFile( $summary,          \\%headerText,       \\%tsv );

sub writeFile {
    my $summary    = $_[0];
    my %headerText = %{ $_[1] };
    my %tsv        = %{ $_[2] };
    open( OUT, ">$summary" );
    print OUT "Sample\\t";
    my @headArr = ();
    for my $mapOrder (@sortedOrderArray) {
        push( @headArr, @{ $headerText{$mapOrder} } );
    }
    my $headArrAll = join( "\\t", @headArr );
    print OUT "$headArrAll\\n";

    foreach my $name ( keys %tsv ) {
        my @rowArr = ();
        for my $mapOrder (@sortedOrderArray) {
            push( @rowArr, @{ $tsv{$name}{ $headerHash{$mapOrder} } } );
        }
        my $rowArrAll = join( "\\t", @rowArr );
        print OUT "$name\\t$rowArrAll\\n";
    }
    close(OUT);
}

'''
}


process Adapter_Trimmer_Quality_Module_Umitools_Summary {

input:
 file logfile from g257_23_log_file10_g257_24.collect()
 val mate from g_347_mate11_g257_24

output:
 file "umitools_summary.tsv"  into g257_24_outputFileTSV08_g_198

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;

my @header;
my %all_files;
my %tsv;
my %tsvDetail;
my %headerHash;
my %headerText;
my %headerTextDetail;

my $i = 0;
chomp( my $contents = `ls *.log` );

my @files = split( /[\\n]+/, $contents );
foreach my $file (@files) {
    $i++;
    my $mapOrder = "1";
    if ($file =~ /(.*)\\.umitools\\.log/){
        $file =~ /(.*)\\.umitools\\.log/;
        my $mapper   = "umitools";
        my $name = $1;    ##sample name
        push( @header, $mapper );
        my $in;
        my $out;
        my $dedupout;
        chomp( $in =`cat $file | grep 'INFO Input Reads:' | awk '{sum=\\$6} END {print sum}'` );
        chomp( $out =`cat $file | grep 'INFO Reads output:' | awk '{sum=\\$6} END {print sum}'` );
        my $deduplog = $name.".dedup.log";
        $headerHash{$mapOrder} = $mapper;
        if (-e $deduplog) {
            print "dedup log found\\n";
            chomp( $dedupout =`cat $deduplog | grep '$name' | awk '{sum=\\$3} END {print sum}'` );
            $tsv{$name}{$mapper} = [ $in, $out, $dedupout];
            $headerText{$mapOrder} = [ "Total Reads", "Reads After Umiextract", "Reads After Deduplication" ]; 
        } else {
            $tsv{$name}{$mapper} = [ $in, $out ];
            $headerText{$mapOrder} = [ "Total Reads", "Reads After Umiextract" ]; 
        }
        
        
    }
}

my @mapOrderArray = ( keys %headerHash );
my @sortedOrderArray = sort { $a <=> $b } @mapOrderArray;

my $summary          = "umitools_summary.tsv";
writeFile( $summary,          \\%headerText,       \\%tsv );

sub writeFile {
    my $summary    = $_[0];
    my %headerText = %{ $_[1] };
    my %tsv        = %{ $_[2] };
    open( OUT, ">$summary" );
    print OUT "Sample\\t";
    my @headArr = ();
    for my $mapOrder (@sortedOrderArray) {
        push( @headArr, @{ $headerText{$mapOrder} } );
    }
    my $headArrAll = join( "\\t", @headArr );
    print OUT "$headArrAll\\n";

    foreach my $name ( keys %tsv ) {
        my @rowArr = ();
        for my $mapOrder (@sortedOrderArray) {
            push( @rowArr, @{ $tsv{$name}{ $headerHash{$mapOrder} } } );
        }
        my $rowArrAll = join( "\\t", @rowArr );
        print OUT "$name\\t$rowArrAll\\n";
    }
    close(OUT);
}

'''
}


process STAR_Module_STAR_Summary {

input:
 set val(name), file(alignSum) from g264_31_outputFileOut00_g264_18.groupTuple()

output:
 file "*.tsv"  into g264_18_outputFile00_g264_11
 val "star_alignment_sum"  into g264_18_name11_g264_11

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage; 
use Data::Dumper;

my %tsv;
my @headers = ();
my $name = '!{name}';


alteredAligned();

my @keys = keys %tsv;
my $summary = "$name"."_star_sum.tsv";
my $header_string = join("\\t", @headers);
`echo "$header_string" > $summary`;
foreach my $key (@keys){
	my $values = join("\\t", @{ $tsv{$key} });
	`echo "$values" >> $summary`;
}


sub alteredAligned
{
	my @files = qw(!{alignSum});
	my $multimappedSum;
	my $alignedSum;
	my $inputCountSum;
	push(@headers, "Sample");
    push(@headers, "Total Reads");
	push(@headers, "Multimapped Reads Aligned (STAR)");
	push(@headers, "Unique Reads Aligned (STAR)");
	foreach my $file (@files){
		my $multimapped;
		my $aligned;
		my $inputCount;
		chomp($inputCount = `cat $file | grep 'Number of input reads' | awk '{sum+=\\$6} END {print sum}'`);
		chomp($aligned = `cat $file | grep 'Uniquely mapped reads number' | awk '{sum+=\\$6} END {print sum}'`);
		chomp($multimapped = `cat $file | grep 'Number of reads mapped to multiple loci' | awk '{sum+=\\$9} END {print sum}'`);
		$multimappedSum += int($multimapped);
        $alignedSum += int($aligned);
        $inputCountSum += int($inputCount);
	}
	$tsv{$name} = [$name, $inputCountSum];
	push(@{$tsv{$name}}, $multimappedSum);
	push(@{$tsv{$name}}, $alignedSum);
}

sub runCommand {
    my ($com) = @_;
    if ($com eq ""){
		return "";
    }
    my $error = system(@_);
    if   ($error) { die "Command failed: $error $com\\n"; }
    else          { print "Command successful: $com\\n"; }
}
'''

}


process STAR_Module_merge_tsv_files_with_same_header {

input:
 file tsv from g264_18_outputFile00_g264_11.collect()
 val outputFileName from g264_18_name11_g264_11.collect()

output:
 file "${name}.tsv"  into g264_11_outputFileTSV00_g_198

errorStrategy 'retry'
maxRetries 3

script:
name = outputFileName[0]
"""    
awk 'FNR==1 && NR!=1 {  getline; } 1 {print} ' *.tsv > ${name}.tsv
"""
}

g264_11_outputFileTSV00_g_198= g264_11_outputFileTSV00_g_198.ifEmpty([""]) 
g256_14_outputFileTSV01_g_198= g256_14_outputFileTSV01_g_198.ifEmpty([""]) 
g249_10_outputFileTSV02_g_198= g249_10_outputFileTSV02_g_198.ifEmpty([""]) 
g250_17_outputFileTSV03_g_198= g250_17_outputFileTSV03_g_198.ifEmpty([""]) 
g257_11_outputFileTSV05_g_198= g257_11_outputFileTSV05_g_198.ifEmpty([""]) 
g257_21_outputFileTSV06_g_198= g257_21_outputFileTSV06_g_198.ifEmpty([""]) 
g257_16_outputFileTSV07_g_198= g257_16_outputFileTSV07_g_198.ifEmpty([""]) 
g257_24_outputFileTSV08_g_198= g257_24_outputFileTSV08_g_198.ifEmpty([""]) 
g248_22_outFileTSV09_g_198= g248_22_outFileTSV09_g_198.ifEmpty([""]) 
g268_45_outFileTSV011_g_198= g268_45_outFileTSV011_g_198.ifEmpty([""]) 

//* autofill
//* platform
if ($HOSTNAME == "ghpcc06.umassrc.org"){
    $TIME = 30
    $CPU  = 1
    $MEMORY = 10
    $QUEUE = "short"
}
//* platform
//* autofill

process Overall_Summary {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /overall_summary.tsv$/) "summary/$filename"}
input:
 file starSum from g264_11_outputFileTSV00_g_198
 file sequentialSum from g256_14_outputFileTSV01_g_198
 file hisatSum from g249_10_outputFileTSV02_g_198
 file rsemSum from g250_17_outputFileTSV03_g_198
 file adapterSum from g257_11_outputFileTSV05_g_198
 file trimmerSum from g257_21_outputFileTSV06_g_198
 file qualitySum from g257_16_outputFileTSV07_g_198
 file umiSum from g257_24_outputFileTSV08_g_198
 file kallistoSum from g248_22_outFileTSV09_g_198
 file salmonSum from g268_45_outFileTSV011_g_198

output:
 file "overall_summary.tsv" optional true  into g_198_outputFileTSV00

shell:
'''
#!/usr/bin/env perl
use List::Util qw[min max];
use strict;
use File::Basename;
use Getopt::Long;
use Pod::Usage;
use Data::Dumper;

my @header;
my %all_rows;
my @seen_cols;
my $ID_header;

chomp(my $contents = `ls *.tsv`);
my @rawFiles = split(/[\\n]+/, $contents);
if (scalar @rawFiles == 0){
    exit;
}
my @files = ();
# order must be in this order for chipseq pipeline: bowtie->dedup
# rsem bam pipeline: dedup->rsem, star->dedup
# riboseq ncRNA_removal->star
my @order = ("adapter_removal","trimmer","quality","extractUMI","extractValid","tRAX","sequential_mapping","ncRNA_removal","bowtie","star","hisat2","tophat2", "dedup","rsem","kallisto","salmon","esat","count");
for ( my $k = 0 ; $k <= $#order ; $k++ ) {
    for ( my $i = 0 ; $i <= $#rawFiles ; $i++ ) {
        if ( $rawFiles[$i] =~ /$order[$k]/ ) {
            push @files, $rawFiles[$i];
        }
    }
}

print Dumper \\@files;
##add rest of the files
for ( my $i = 0 ; $i <= $#rawFiles ; $i++ ) {
    push(@files, $rawFiles[$i]) unless grep{$_ == $rawFiles[$i]} @files;
}
print Dumper \\@files;

##Merge each file according to array order

foreach my $file (@files){
        open IN,"$file";
        my $line1 = <IN>;
        chomp($line1);
        ( $ID_header, my @header) = ( split("\\t", $line1) );
        push @seen_cols, @header;

        while (my $line=<IN>) {
        chomp($line);
        my ( $ID, @fields ) = ( split("\\t", $line) ); 
        my %this_row;
        @this_row{@header} = @fields;

        #print Dumper \\%this_row;

        foreach my $column (@header) {
            if (! exists $all_rows{$ID}{$column}) {
                $all_rows{$ID}{$column} = $this_row{$column}; 
            }
        }   
    }
    close IN;
}

#print for debugging
#print Dumper \\%all_rows;
#print Dumper \\%seen_cols;

#grab list of column headings we've seen, and order them. 
my @cols_to_print = uniq(@seen_cols);
my $summary = "overall_summary.tsv";
open OUT, ">$summary";
print OUT join ("\\t", $ID_header,@cols_to_print),"\\n";
foreach my $key ( keys %all_rows ) { 
    #map iterates all the columns, and gives the value or an empty string. if it's undefined. (prevents errors)
    print OUT join ("\\t", $key, (map { $all_rows{$key}{$_} // '' } @cols_to_print)),"\\n";
}
close OUT;

sub uniq {
    my %seen;
    grep ! $seen{$_}++, @_;
}

'''


}

g292_24_postfix10_g292_33= g292_24_postfix10_g292_33.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_RSEM_Prepare_GSEA_DESeq2 {

input:
 val postfix from g292_24_postfix10_g292_33
 file input from g292_24_outputFile21_g292_33
 val run_GSEA from g_375_2_g292_33

output:
 file "GSEA_reports"  into g292_33_outputFile01_g292_37
 file "GSEA"  into g292_33_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_RSEM_Prepare_GSEA_DESeq2.event_column
fold_change_column = params.DE_module_RSEM_Prepare_GSEA_DESeq2.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_RSEM_Prepare_GSEA_DESeq2.local_species
H = params.DE_module_RSEM_Prepare_GSEA_DESeq2.H
C1 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C1
C2 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C2
C2_CGP = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C2_CGP
C2_CP = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C2_CP
C3_MIR = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C3_MIR
C3_TFT = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C3_TFT
C4 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C4
C4_CGN = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C4_CGN
C4_CM = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C4_CM
C5 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C5
C5_GO = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C5_GO
C5_GO_BP = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C5_GO_BP
C5_GO_CC = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C5_GO_CC
C5_GO_MF = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C5_GO_MF
C5_HPO = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C5_HPO
C6 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C6
C7 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C7
C8 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.C8
MH = params.DE_module_RSEM_Prepare_GSEA_DESeq2.MH
M1 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M1
M2 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M2
M2_CGP = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M2_CGP
M2_CP = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M2_CP
M3_GTRD = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M3_GTRD
M3_miRDB = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M3_miRDB
M5 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M5
M5_GO = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M5_GO
M5_GO_BP = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M5_GO_BP
M5_GO_CC = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M5_GO_CC
M5_GO_MF = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M5_GO_MF
M5_MPT = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M5_MPT
M8 = params.DE_module_RSEM_Prepare_GSEA_DESeq2.M8

minSize = params.DE_module_RSEM_Prepare_GSEA_DESeq2.minSize
maxSize = params.DE_module_RSEM_Prepare_GSEA_DESeq2.maxSize

nes_significance_cutoff = params.DE_module_RSEM_Prepare_GSEA_DESeq2.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_RSEM_Prepare_GSEA_DESeq2.padj_significance_cutoff

seed = params.DE_module_RSEM_Prepare_GSEA_DESeq2.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g292_33_outputFile01_g292_37= g292_33_outputFile01_g292_37.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_RSEM_DESeq2_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "DESeq2_RSEM/$filename"}
input:
 file DE_reports from g292_24_outputFile00_g292_37
 file GSEA_reports from g292_33_outputFile01_g292_37

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g292_37_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g292_25_postfix10_g292_41= g292_25_postfix10_g292_41.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_RSEM_Prepare_GSEA_LimmaVoom {

input:
 val postfix from g292_25_postfix10_g292_41
 file input from g292_25_outputFile21_g292_41
 val run_GSEA from g_376_2_g292_41

output:
 file "GSEA_reports"  into g292_41_outputFile01_g292_39
 file "GSEA"  into g292_41_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.event_column
fold_change_column = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.local_species
H = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.H
C1 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C1
C2 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C2
C2_CGP = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C2_CGP
C2_CP = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C2_CP
C3_MIR = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C3_MIR
C3_TFT = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C3_TFT
C4 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C4
C4_CGN = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C4_CGN
C4_CM = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C4_CM
C5 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C5
C5_GO = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C5_GO
C5_GO_BP = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C5_GO_BP
C5_GO_CC = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C5_GO_CC
C5_GO_MF = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C5_GO_MF
C5_HPO = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C5_HPO
C6 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C6
C7 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C7
C8 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.C8
MH = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.MH
M1 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M1
M2 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M2
M2_CGP = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M2_CGP
M2_CP = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M2_CP
M3_GTRD = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M3_GTRD
M3_miRDB = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M3_miRDB
M5 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M5
M5_GO = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M5_GO
M5_GO_BP = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M5_GO_BP
M5_GO_CC = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M5_GO_CC
M5_GO_MF = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M5_GO_MF
M5_MPT = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M5_MPT
M8 = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.M8

minSize = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.minSize
maxSize = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.maxSize

nes_significance_cutoff = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.padj_significance_cutoff

seed = params.DE_module_RSEM_Prepare_GSEA_LimmaVoom.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g292_41_outputFile01_g292_39= g292_41_outputFile01_g292_39.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_RSEM_LimmaVoom_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "limmaVoom_RSEM/$filename"}
input:
 file DE_reports from g292_25_outputFile00_g292_39
 file GSEA_reports from g292_41_outputFile01_g292_39

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g292_39_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g302_24_postfix10_g302_33= g302_24_postfix10_g302_33.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_Salmon_Prepare_GSEA_DESeq2 {

input:
 val postfix from g302_24_postfix10_g302_33
 file input from g302_24_outputFile21_g302_33
 val run_GSEA from g_393_2_g302_33

output:
 file "GSEA_reports"  into g302_33_outputFile01_g302_37
 file "GSEA"  into g302_33_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.event_column
fold_change_column = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.local_species
H = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.H
C1 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C1
C2 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C2
C2_CGP = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C2_CGP
C2_CP = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C2_CP
C3_MIR = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C3_MIR
C3_TFT = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C3_TFT
C4 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C4
C4_CGN = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C4_CGN
C4_CM = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C4_CM
C5 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C5
C5_GO = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C5_GO
C5_GO_BP = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C5_GO_BP
C5_GO_CC = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C5_GO_CC
C5_GO_MF = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C5_GO_MF
C5_HPO = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C5_HPO
C6 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C6
C7 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C7
C8 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.C8
MH = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.MH
M1 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M1
M2 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M2
M2_CGP = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M2_CGP
M2_CP = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M2_CP
M3_GTRD = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M3_GTRD
M3_miRDB = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M3_miRDB
M5 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M5
M5_GO = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M5_GO
M5_GO_BP = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M5_GO_BP
M5_GO_CC = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M5_GO_CC
M5_GO_MF = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M5_GO_MF
M5_MPT = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M5_MPT
M8 = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.M8

minSize = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.minSize
maxSize = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.maxSize

nes_significance_cutoff = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.padj_significance_cutoff

seed = params.DE_module_STAR_Salmon_Prepare_GSEA_DESeq2.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g302_33_outputFile01_g302_37= g302_33_outputFile01_g302_37.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_Salmon_DESeq2_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "DESeq2_STAR_Salmon/$filename"}
input:
 file DE_reports from g302_24_outputFile00_g302_37
 file GSEA_reports from g302_33_outputFile01_g302_37

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g302_37_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g302_25_postfix10_g302_41= g302_25_postfix10_g302_41.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom {

input:
 val postfix from g302_25_postfix10_g302_41
 file input from g302_25_outputFile21_g302_41
 val run_GSEA from g_394_2_g302_41

output:
 file "GSEA_reports"  into g302_41_outputFile01_g302_39
 file "GSEA"  into g302_41_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.event_column
fold_change_column = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.local_species
H = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.H
C1 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C1
C2 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C2
C2_CGP = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C2_CGP
C2_CP = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C2_CP
C3_MIR = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C3_MIR
C3_TFT = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C3_TFT
C4 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C4
C4_CGN = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C4_CGN
C4_CM = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C4_CM
C5 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C5
C5_GO = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C5_GO
C5_GO_BP = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C5_GO_BP
C5_GO_CC = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C5_GO_CC
C5_GO_MF = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C5_GO_MF
C5_HPO = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C5_HPO
C6 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C6
C7 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C7
C8 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.C8
MH = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.MH
M1 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M1
M2 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M2
M2_CGP = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M2_CGP
M2_CP = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M2_CP
M3_GTRD = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M3_GTRD
M3_miRDB = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M3_miRDB
M5 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M5
M5_GO = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M5_GO
M5_GO_BP = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M5_GO_BP
M5_GO_CC = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M5_GO_CC
M5_GO_MF = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M5_GO_MF
M5_MPT = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M5_MPT
M8 = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.M8

minSize = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.minSize
maxSize = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.maxSize

nes_significance_cutoff = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.padj_significance_cutoff

seed = params.DE_module_STAR_Salmon_Prepare_GSEA_LimmaVoom.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g302_41_outputFile01_g302_39= g302_41_outputFile01_g302_39.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_Salmon_LimmaVoom_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "limmaVoom_STAR_Salmon/$filename"}
input:
 file DE_reports from g302_25_outputFile00_g302_39
 file GSEA_reports from g302_41_outputFile01_g302_39

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g302_39_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g303_24_postfix10_g303_33= g303_24_postfix10_g303_33.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_Kallisto_Prepare_GSEA_DESeq2 {

input:
 val postfix from g303_24_postfix10_g303_33
 file input from g303_24_outputFile21_g303_33
 val run_GSEA from g_395_2_g303_33

output:
 file "GSEA_reports"  into g303_33_outputFile01_g303_37
 file "GSEA"  into g303_33_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.event_column
fold_change_column = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.local_species
H = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.H
C1 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C1
C2 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C2
C2_CGP = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C2_CGP
C2_CP = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C2_CP
C3_MIR = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C3_MIR
C3_TFT = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C3_TFT
C4 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C4
C4_CGN = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C4_CGN
C4_CM = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C4_CM
C5 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C5
C5_GO = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C5_GO
C5_GO_BP = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C5_GO_BP
C5_GO_CC = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C5_GO_CC
C5_GO_MF = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C5_GO_MF
C5_HPO = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C5_HPO
C6 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C6
C7 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C7
C8 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.C8
MH = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.MH
M1 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M1
M2 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M2
M2_CGP = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M2_CGP
M2_CP = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M2_CP
M3_GTRD = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M3_GTRD
M3_miRDB = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M3_miRDB
M5 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M5
M5_GO = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M5_GO
M5_GO_BP = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M5_GO_BP
M5_GO_CC = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M5_GO_CC
M5_GO_MF = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M5_GO_MF
M5_MPT = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M5_MPT
M8 = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.M8

minSize = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.minSize
maxSize = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.maxSize

nes_significance_cutoff = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.padj_significance_cutoff

seed = params.DE_module_Kallisto_Prepare_GSEA_DESeq2.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g303_33_outputFile01_g303_37= g303_33_outputFile01_g303_37.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_Kallisto_DESeq2_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "DESeq2_Kallisto/$filename"}
input:
 file DE_reports from g303_24_outputFile00_g303_37
 file GSEA_reports from g303_33_outputFile01_g303_37

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g303_37_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g303_25_postfix10_g303_41= g303_25_postfix10_g303_41.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_Kallisto_Prepare_GSEA_LimmaVoom {

input:
 val postfix from g303_25_postfix10_g303_41
 file input from g303_25_outputFile21_g303_41
 val run_GSEA from g_396_2_g303_41

output:
 file "GSEA_reports"  into g303_41_outputFile01_g303_39
 file "GSEA"  into g303_41_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.event_column
fold_change_column = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.local_species
H = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.H
C1 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C1
C2 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C2
C2_CGP = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C2_CGP
C2_CP = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C2_CP
C3_MIR = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C3_MIR
C3_TFT = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C3_TFT
C4 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C4
C4_CGN = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C4_CGN
C4_CM = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C4_CM
C5 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C5
C5_GO = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C5_GO
C5_GO_BP = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C5_GO_BP
C5_GO_CC = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C5_GO_CC
C5_GO_MF = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C5_GO_MF
C5_HPO = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C5_HPO
C6 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C6
C7 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C7
C8 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.C8
MH = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.MH
M1 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M1
M2 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M2
M2_CGP = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M2_CGP
M2_CP = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M2_CP
M3_GTRD = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M3_GTRD
M3_miRDB = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M3_miRDB
M5 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M5
M5_GO = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M5_GO
M5_GO_BP = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M5_GO_BP
M5_GO_CC = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M5_GO_CC
M5_GO_MF = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M5_GO_MF
M5_MPT = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M5_MPT
M8 = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.M8

minSize = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.minSize
maxSize = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.maxSize

nes_significance_cutoff = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.padj_significance_cutoff

seed = params.DE_module_Kallisto_Prepare_GSEA_LimmaVoom.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g303_41_outputFile01_g303_39= g303_41_outputFile01_g303_39.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_Kallisto_LimmaVoom_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "limmaVoom_Kallisto/$filename"}
input:
 file DE_reports from g303_25_outputFile00_g303_39
 file GSEA_reports from g303_41_outputFile01_g303_39

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g303_39_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g304_24_postfix10_g304_33= g304_24_postfix10_g304_33.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_Salmon_Prepare_GSEA_DESeq2 {

input:
 val postfix from g304_24_postfix10_g304_33
 file input from g304_24_outputFile21_g304_33
 val run_GSEA from g_397_2_g304_33

output:
 file "GSEA_reports"  into g304_33_outputFile01_g304_37
 file "GSEA"  into g304_33_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_Salmon_Prepare_GSEA_DESeq2.event_column
fold_change_column = params.DE_module_Salmon_Prepare_GSEA_DESeq2.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_Salmon_Prepare_GSEA_DESeq2.local_species
H = params.DE_module_Salmon_Prepare_GSEA_DESeq2.H
C1 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C1
C2 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C2
C2_CGP = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C2_CGP
C2_CP = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C2_CP
C3_MIR = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C3_MIR
C3_TFT = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C3_TFT
C4 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C4
C4_CGN = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C4_CGN
C4_CM = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C4_CM
C5 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C5
C5_GO = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C5_GO
C5_GO_BP = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C5_GO_BP
C5_GO_CC = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C5_GO_CC
C5_GO_MF = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C5_GO_MF
C5_HPO = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C5_HPO
C6 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C6
C7 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C7
C8 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.C8
MH = params.DE_module_Salmon_Prepare_GSEA_DESeq2.MH
M1 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M1
M2 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M2
M2_CGP = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M2_CGP
M2_CP = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M2_CP
M3_GTRD = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M3_GTRD
M3_miRDB = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M3_miRDB
M5 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M5
M5_GO = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M5_GO
M5_GO_BP = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M5_GO_BP
M5_GO_CC = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M5_GO_CC
M5_GO_MF = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M5_GO_MF
M5_MPT = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M5_MPT
M8 = params.DE_module_Salmon_Prepare_GSEA_DESeq2.M8

minSize = params.DE_module_Salmon_Prepare_GSEA_DESeq2.minSize
maxSize = params.DE_module_Salmon_Prepare_GSEA_DESeq2.maxSize

nes_significance_cutoff = params.DE_module_Salmon_Prepare_GSEA_DESeq2.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_Salmon_Prepare_GSEA_DESeq2.padj_significance_cutoff

seed = params.DE_module_Salmon_Prepare_GSEA_DESeq2.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g304_33_outputFile01_g304_37= g304_33_outputFile01_g304_37.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_Salmon_DESeq2_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "DESeq2_Salmon/$filename"}
input:
 file DE_reports from g304_24_outputFile00_g304_37
 file GSEA_reports from g304_33_outputFile01_g304_37

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g304_37_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g304_25_postfix10_g304_41= g304_25_postfix10_g304_41.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_Salmon_Prepare_GSEA_LimmaVoom {

input:
 val postfix from g304_25_postfix10_g304_41
 file input from g304_25_outputFile21_g304_41
 val run_GSEA from g_398_2_g304_41

output:
 file "GSEA_reports"  into g304_41_outputFile01_g304_39
 file "GSEA"  into g304_41_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.event_column
fold_change_column = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.local_species
H = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.H
C1 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C1
C2 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C2
C2_CGP = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C2_CGP
C2_CP = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C2_CP
C3_MIR = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C3_MIR
C3_TFT = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C3_TFT
C4 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C4
C4_CGN = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C4_CGN
C4_CM = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C4_CM
C5 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C5
C5_GO = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C5_GO
C5_GO_BP = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C5_GO_BP
C5_GO_CC = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C5_GO_CC
C5_GO_MF = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C5_GO_MF
C5_HPO = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C5_HPO
C6 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C6
C7 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C7
C8 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.C8
MH = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.MH
M1 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M1
M2 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M2
M2_CGP = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M2_CGP
M2_CP = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M2_CP
M3_GTRD = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M3_GTRD
M3_miRDB = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M3_miRDB
M5 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M5
M5_GO = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M5_GO
M5_GO_BP = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M5_GO_BP
M5_GO_CC = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M5_GO_CC
M5_GO_MF = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M5_GO_MF
M5_MPT = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M5_MPT
M8 = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.M8

minSize = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.minSize
maxSize = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.maxSize

nes_significance_cutoff = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.padj_significance_cutoff

seed = params.DE_module_Salmon_Prepare_GSEA_LimmaVoom.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g304_41_outputFile01_g304_39= g304_41_outputFile01_g304_39.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_Salmon_LimmaVoom_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "limmaVoom_Salmon/$filename"}
input:
 file DE_reports from g304_25_outputFile00_g304_39
 file GSEA_reports from g304_41_outputFile01_g304_39

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g304_39_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g305_24_postfix10_g305_33= g305_24_postfix10_g305_33.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2 {

input:
 val postfix from g305_24_postfix10_g305_33
 file input from g305_24_outputFile21_g305_33
 val run_GSEA from g_391_2_g305_33

output:
 file "GSEA_reports"  into g305_33_outputFile01_g305_37
 file "GSEA"  into g305_33_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.event_column
fold_change_column = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.local_species
H = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.H
C1 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C1
C2 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C2
C2_CGP = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C2_CGP
C2_CP = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C2_CP
C3_MIR = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C3_MIR
C3_TFT = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C3_TFT
C4 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C4
C4_CGN = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C4_CGN
C4_CM = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C4_CM
C5 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C5
C5_GO = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C5_GO
C5_GO_BP = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C5_GO_BP
C5_GO_CC = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C5_GO_CC
C5_GO_MF = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C5_GO_MF
C5_HPO = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C5_HPO
C6 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C6
C7 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C7
C8 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.C8
MH = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.MH
M1 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M1
M2 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M2
M2_CGP = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M2_CGP
M2_CP = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M2_CP
M3_GTRD = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M3_GTRD
M3_miRDB = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M3_miRDB
M5 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M5
M5_GO = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M5_GO
M5_GO_BP = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M5_GO_BP
M5_GO_CC = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M5_GO_CC
M5_GO_MF = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M5_GO_MF
M5_MPT = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M5_MPT
M8 = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.M8

minSize = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.minSize
maxSize = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.maxSize

nes_significance_cutoff = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.padj_significance_cutoff

seed = params.DE_module_STAR_featurecounts_Prepare_GSEA_DESeq2.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g305_33_outputFile01_g305_37= g305_33_outputFile01_g305_37.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_featurecounts_DESeq2_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "DESeq2_STAR_featurecounts/$filename"}
input:
 file DE_reports from g305_24_outputFile00_g305_37
 file GSEA_reports from g305_33_outputFile01_g305_37

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g305_37_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g305_25_postfix10_g305_41= g305_25_postfix10_g305_41.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom {

input:
 val postfix from g305_25_postfix10_g305_41
 file input from g305_25_outputFile21_g305_41
 val run_GSEA from g_392_2_g305_41

output:
 file "GSEA_reports"  into g305_41_outputFile01_g305_39
 file "GSEA"  into g305_41_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.event_column
fold_change_column = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.local_species
H = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.H
C1 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C1
C2 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C2
C2_CGP = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C2_CGP
C2_CP = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C2_CP
C3_MIR = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C3_MIR
C3_TFT = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C3_TFT
C4 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C4
C4_CGN = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C4_CGN
C4_CM = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C4_CM
C5 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C5
C5_GO = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C5_GO
C5_GO_BP = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C5_GO_BP
C5_GO_CC = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C5_GO_CC
C5_GO_MF = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C5_GO_MF
C5_HPO = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C5_HPO
C6 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C6
C7 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C7
C8 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.C8
MH = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.MH
M1 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M1
M2 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M2
M2_CGP = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M2_CGP
M2_CP = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M2_CP
M3_GTRD = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M3_GTRD
M3_miRDB = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M3_miRDB
M5 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M5
M5_GO = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M5_GO
M5_GO_BP = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M5_GO_BP
M5_GO_CC = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M5_GO_CC
M5_GO_MF = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M5_GO_MF
M5_MPT = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M5_MPT
M8 = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.M8

minSize = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.minSize
maxSize = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.maxSize

nes_significance_cutoff = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.padj_significance_cutoff

seed = params.DE_module_STAR_featurecounts_Prepare_GSEA_LimmaVoom.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g305_41_outputFile01_g305_39= g305_41_outputFile01_g305_39.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_STAR_featurecounts_LimmaVoom_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "limmaVoom_STAR_featurecounts/$filename"}
input:
 file DE_reports from g305_25_outputFile00_g305_39
 file GSEA_reports from g305_41_outputFile01_g305_39

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g305_39_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g306_24_postfix10_g306_33= g306_24_postfix10_g306_33.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2 {

input:
 val postfix from g306_24_postfix10_g306_33
 file input from g306_24_outputFile21_g306_33
 val run_GSEA from g_389_2_g306_33

output:
 file "GSEA_reports"  into g306_33_outputFile01_g306_37
 file "GSEA"  into g306_33_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.event_column
fold_change_column = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.local_species
H = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.H
C1 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C1
C2 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C2
C2_CGP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C2_CGP
C2_CP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C2_CP
C3_MIR = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C3_MIR
C3_TFT = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C3_TFT
C4 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C4
C4_CGN = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C4_CGN
C4_CM = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C4_CM
C5 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C5
C5_GO = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C5_GO
C5_GO_BP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C5_GO_BP
C5_GO_CC = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C5_GO_CC
C5_GO_MF = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C5_GO_MF
C5_HPO = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C5_HPO
C6 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C6
C7 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C7
C8 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.C8
MH = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.MH
M1 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M1
M2 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M2
M2_CGP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M2_CGP
M2_CP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M2_CP
M3_GTRD = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M3_GTRD
M3_miRDB = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M3_miRDB
M5 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M5
M5_GO = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M5_GO
M5_GO_BP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M5_GO_BP
M5_GO_CC = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M5_GO_CC
M5_GO_MF = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M5_GO_MF
M5_MPT = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M5_MPT
M8 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.M8

minSize = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.minSize
maxSize = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.maxSize

nes_significance_cutoff = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.padj_significance_cutoff

seed = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_DESeq2.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g306_33_outputFile01_g306_37= g306_33_outputFile01_g306_37.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_HISAT2_featurecounts_DESeq2_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "DESeq2_HISAT2_featurecounts/$filename"}
input:
 file DE_reports from g306_24_outputFile00_g306_37
 file GSEA_reports from g306_33_outputFile01_g306_37

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g306_37_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}

g306_25_postfix10_g306_41= g306_25_postfix10_g306_41.ifEmpty("") 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 5
    $MEMORY = 30 
}
//* platform
//* platform
//* autofill

process DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom {

input:
 val postfix from g306_25_postfix10_g306_41
 file input from g306_25_outputFile21_g306_41
 val run_GSEA from g_390_2_g306_41

output:
 file "GSEA_reports"  into g306_41_outputFile01_g306_39
 file "GSEA"  into g306_41_outputFile11

container 'quay.io/viascientific/gsea_module:1.0.0'

// SET second output to "{GSEA,outputs}" when launched apps can reach parent directory

when:
run_GSEA == 'yes'

script:

if (postfix.equals(" ")) {
	postfix = ''
}

event_column = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.event_column
fold_change_column = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.fold_change_column

if (params.genome_build.substring(0,5).equals("human")) {
	species = 'human'
} else if (params.genome_build.substring(0,5).equals("mouse")) {
	species = 'mouse'
} else {
	species = 'NA'
}

local_species = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.local_species
H = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.H
C1 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C1
C2 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C2
C2_CGP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C2_CGP
C2_CP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C2_CP
C3_MIR = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C3_MIR
C3_TFT = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C3_TFT
C4 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C4
C4_CGN = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C4_CGN
C4_CM = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C4_CM
C5 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C5
C5_GO = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C5_GO
C5_GO_BP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C5_GO_BP
C5_GO_CC = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C5_GO_CC
C5_GO_MF = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C5_GO_MF
C5_HPO = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C5_HPO
C6 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C6
C7 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C7
C8 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.C8
MH = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.MH
M1 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M1
M2 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M2
M2_CGP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M2_CGP
M2_CP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M2_CP
M3_GTRD = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M3_GTRD
M3_miRDB = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M3_miRDB
M5 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M5
M5_GO = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M5_GO
M5_GO_BP = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M5_GO_BP
M5_GO_CC = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M5_GO_CC
M5_GO_MF = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M5_GO_MF
M5_MPT = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M5_MPT
M8 = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.M8

minSize = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.minSize
maxSize = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.maxSize

nes_significance_cutoff = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.nes_significance_cutoff
padj_significance_cutoff = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.padj_significance_cutoff

seed = params.DE_module_HISAT2_featurecounts_Prepare_GSEA_LimmaVoom.seed

//* @style @condition:{local_species="human",H,C1,C2,C2_CGP,C2_CP,C3_MIR,C3_TFT,C4,C4_CGN,C4_CM,C5,C5_GO,C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO,C6,C7,C8},{local_species="mouse",MH,M1,M2,M2_CGP,M2_CP,M3_GTRD,M3_miRDB,M5,M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF,M5_MPT,M8} @multicolumn:{event_column, fold_change_column}, {gmt_list, minSize, maxSize},{H,C1,C2,C2_CGP}, {C2_CP,C3_MIR,C3_TFT,C4},{C4_CGN,C4_CM, C5,C5_GO},{C5_GO_BP,C5_GO_CC,C5_GO_MF,C5_HPO},{C6,C7,C8},{MH,M1,M2,M2_CGP},{M2_CP,M3_GTRD,M3_miRDB,M5},{M5_GO,M5_GO_BP,M5_GO_CC,M5_GO_MF},{M5_MPT,M8},{nes_significance_cutoff, padj_significance_cutoff}

H        = H        == 'true' && species == 'human' ? ' h.all.v2023.1.Hs.entrez.gmt'    : ''
C1       = C1       == 'true' && species == 'human' ? ' c1.all.v2023.1.Hs.entrez.gmt'   : ''
C2       = C2       == 'true' && species == 'human' ? ' c2.all.v2023.1.Hs.entrez.gmt'   : ''
C2_CGP   = C2_CGP   == 'true' && species == 'human' ? ' c2.cgp.v2023.1.Hs.entrez.gmt'   : ''
C2_CP    = C2_CP    == 'true' && species == 'human' ? ' c2.cp.v2023.1.Hs.entrez.gmt'    : ''
C3_MIR   = C3_MIR   == 'true' && species == 'human' ? ' c3.mir.v2023.1.Hs.entrez.gmt'   : ''
C3_TFT   = C3_TFT   == 'true' && species == 'human' ? ' c3.tft.v2023.1.Hs.entrez.gmt'   : ''
C4       = C4       == 'true' && species == 'human' ? ' c4.all.v2023.1.Hs.entrez.gmt'   : ''
C4_CGN   = C4_CGN   == 'true' && species == 'human' ? ' c4.cgn.v2023.1.Hs.entrez.gmt'   : ''
C4_CM    = C4_CM    == 'true' && species == 'human' ? ' c4.cm.v2023.1.Hs.entrez.gmt'    : ''
C5       = C5       == 'true' && species == 'human' ? ' c5.all.v2023.1.Hs.entrez.gmt'   : ''
C5_GO    = C5_GO    == 'true' && species == 'human' ? ' c5.go.v2023.1.Hs.entrez.gmt'    : ''
C5_GO_BP = C5_GO_BP == 'true' && species == 'human' ? ' c5.go.bp.v2023.1.Hs.entrez.gmt' : ''
C5_GO_CC = C5_GO_CC == 'true' && species == 'human' ? ' c5.go.cc.v2023.1.Hs.entrez.gmt' : ''
C5_GO_MF = C5_GO_MF == 'true' && species == 'human' ? ' c5.go.mf.v2023.1.Hs.entrez.gmt' : ''
C5_HPO   = C5_HPO   == 'true' && species == 'human' ? ' c5.hpo.v2023.1.Hs.entrez.gmt'   : ''
C6       = C6       == 'true' && species == 'human' ? ' c6.all.v2023.1.Hs.entrez.gmt'   : ''
C7       = C7       == 'true' && species == 'human' ? ' c7.all.v2023.1.Hs.entrez.gmt'   : ''
C8       = C8       == 'true' && species == 'human' ? ' c8.all.v2023.1.Hs.entrez.gmt'   : ''
MH       = MH       == 'true' && species == 'mouse' ? ' mh.all.v2023.1.Mm.entrez.gmt'   : ''    
M1       = M1       == 'true' && species == 'mouse' ? ' m1.all.v2023.1.Mm.entrez.gmt'   : ''    
M2       = M2       == 'true' && species == 'mouse' ? ' m2.all.v2023.1.Mm.entrez.gmt'   : ''    
M2_CGP   = M2_CGP   == 'true' && species == 'mouse' ? ' m2.cgp.v2023.1.Mm.entrez.gmt'   : ''
M2_CP    = M2_CP    == 'true' && species == 'mouse' ? ' m2.cp.v2023.1.Mm.entrez.gmt'    : '' 
M3_GTRD  = M3_GTRD  == 'true' && species == 'mouse' ? ' m3.gtrd.v2023.1.Mm.entrez.gmt'  : ''
M3_miRDB = M3_miRDB == 'true' && species == 'mouse' ? ' m3.mirdb.v2023.1.Mm.entrez.gmt' : ''
M5       = M5       == 'true' && species == 'mouse' ? ' m5.all.v2023.1.Mm.entrez.gmt'   : ''    
M5_GO    = M5_GO    == 'true' && species == 'mouse' ? ' m5.go.v2023.1.Mm.entrez.gmt'    : '' 
M5_GO_BP = M5_GO_BP == 'true' && species == 'mouse' ? ' m5.go.bp.v2023.1.Mm.entrez.gmt' : ''
M5_GO_CC = M5_GO_CC == 'true' && species == 'mouse' ? ' m5.go.cc.v2023.1.Mm.entrez.gmt' : ''
M5_GO_MF = M5_GO_MF == 'true' && species == 'mouse' ? ' m5.go.mf.v2023.1.Mm.entrez.gmt' : ''
M5_MPT   = M5_MPT   == 'true' && species == 'mouse' ? ' m5.mpt.v2023.1.Mm.entrez.gmt'   : ''
M8       = M8       == 'true' && species == 'mouse' ? ' m8.all.v2023.1.Mm.entrez.gmt'   : ''

gmt_list = H + C1 + C2 + C2_CGP + C2_CP + C3_MIR + C3_TFT + C4 + C4_CGN + C4_CM + C5 + C5_GO + C5_GO_BP + C5_GO_CC + C5_GO_MF + C5_HPO + C6 + C7 + C8 + MH + M1 + M2 + M2_CGP + M2_CP + M3_GTRD + M3_miRDB + M5 + M5_GO + M5_GO_BP + M5_GO_CC + M5_GO_MF + M5_MPT + M8

if (gmt_list.equals("")){
	gmt_list = 'h.all.v2023.1.Hs.entrez.gmt'
}

"""
prepare_GSEA.py \
--input ${input} --species ${species} --event-column ${event_column} --fold-change-column ${fold_change_column} \
--GMT-key /data/gmt_key.txt --GMT-source /data --GMT-list ${gmt_list} --minSize ${minSize} --maxSize ${maxSize} \
--NES ${nes_significance_cutoff} --pvalue ${padj_significance_cutoff} \
--seed ${seed} --threads ${task.cpus} --postfix '_gsea${postfix}'

cp -R outputs GSEA/

mkdir GSEA_reports
cp -R GSEA GSEA_reports/
"""

}

g306_41_outputFile01_g306_39= g306_41_outputFile01_g306_39.ifEmpty([""]) 

//* autofill
if ($HOSTNAME == "default"){
    $CPU  = 1
    $MEMORY = 4 
}
//* platform
//* platform
//* autofill

process DE_module_HISAT2_featurecounts_LimmaVoom_Analysis {

publishDir params.outdir, mode: 'copy', saveAs: {filename -> if (filename =~ /(.*.Rmd|.*.html|inputs|outputs|GSEA)$/) "limmaVoom_HISAT2_featurecounts/$filename"}
input:
 file DE_reports from g306_25_outputFile00_g306_39
 file GSEA_reports from g306_41_outputFile01_g306_39

output:
 file "{*.Rmd,*.html,inputs,outputs,GSEA}"  into g306_39_outputDir00

script:

"""
mv DE_reports/* .

if [ -d "GSEA_reports" ]; then
    mv GSEA_reports/* .
fi
"""
}


workflow.onComplete {
println "##Pipeline execution summary##"
println "---------------------------"
println "##Completed at: $workflow.complete"
println "##Duration: ${workflow.duration}"
println "##Success: ${workflow.success ? 'OK' : 'failed' }"
println "##Exit status: ${workflow.exitStatus}"
}
