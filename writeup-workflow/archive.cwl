#!/usr/bin/env cwl-runner
#
# Archive a Project (writeup) submission
#
cwlVersion: v1.0
class: CommandLineTool
baseCommand: challengeutils

hints:
  DockerRequirement:
    dockerPull: sagebionetworks/challengeutils:v4.0.1

requirements:
  - class: InlineJavascriptRequirement

inputs:
  - id: submissionid
    type: int
  - id: admin
    type: string
  - id: synapse_config
    type: File
  - id: check_validation_finished
    type: boolean?

arguments:
  - valueFrom: $(inputs.synapse_config.path)
    prefix: -c
  - valueFrom: archive-project
  - valueFrom: $(inputs.submissionid)
  - valueFrom: $(inputs.admin)
  - valueFrom: results.json
    prefix: --output

outputs:
  - id: results
    type: File
    outputBinding:
      glob: results.json