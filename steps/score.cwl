#!/usr/bin/env cwl-runner
cwlVersion: v1.0
class: CommandLineTool
label: Score predictions file

requirements:
- class: InlineJavascriptRequirement

inputs:
- id: input_file
  type: File
- id: goldstandard
  type: File
- id: check_validation_finished
  type: boolean?

outputs:
- id: results
  type: File
  outputBinding:
    glob: results.json
- id: status
  type: string
  outputBinding:
    glob: results.json
    outputEval: $(JSON.parse(self[0].contents)['submission_status'])
    loadContents: true

baseCommand: /usr/local/bin/score.py
arguments:
- prefix: -p
  valueFrom: $(inputs.input_file)
- prefix: -g
  valueFrom: $(inputs.goldstandard.path)
- prefix: -o
  valueFrom: results.json

hints:
  DockerRequirement:
    dockerPull: docker.synapse.org/syn57381674/evaluation:v1

s:author:
- class: s:Person
  s:email: oussama.smaoui@synapse.org
  s:name: Oussama SMAOUI

s:codeRepository: https://github.com/osmaoui/model-to-data-challenge-workflow/tree/main
s:license: https://spdx.org/licenses/Apache-2.0

$namespaces:
  s: https://schema.org/
