#!/usr/bin/env cwl-runner

# INPUTS:
#   submission_id: Submission ID
#   synapse_config: filepath to .synapseConfig file
#   admin_folder_id: Synapse Folder ID accessible by an admin
#   submitter_folder_id: Synapse Folder ID accessible by the submitter
#   workflow_id: Synapse File ID that links to the workflow archive

cwlVersion: v1.0
class: Workflow

label: workflow to accept challenge writeups
doc: >
  This workflow will validate a participant's writeup, checking for:
    - Submission is a Synapse project
    - Submission is not the challenge site (which is a Synapse project)
    - Submission is accessible to the admin team
  Archive (create a project copy) if the submission is valid.

requirements:
  - class: StepInputExpressionRequirement

inputs:
  - id: submissionId
    type: int
  - id: synapseConfig
    type: File
  - id: adminUploadSynId
    type: string
  - id: submitterUploadSynId
    type: string
  - id: workflowSynapseId
    type: string
  - id: admin
    type: string
    default: "jane.doe"  # TODO: enter admin username (they will become the archive owner)

outputs: {}

steps:
  validate:
    run: validate.cwl
    in:
      - id: synapse_config
        source: "#synapseConfig"
      - id: submissionid
        source: "#submissionId"
      - id: challengewiki
        valueFrom: "syn123"  # TODO: update to the Challenge's synID
    # UNCOMMENT THE FOLLOWING IF NEEDED
    #   - id: public
    #     default: true
    #   - id: admin
    #     source: "#admin"
    out:
      - id: results
      - id: status
      - id: invalid_reasons
  
  validation_email:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/validate_email.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: synapse_config
        source: "#synapseConfig"
      - id: status
        source: "#validate/status"
      - id: invalid_reasons
        source: "#validate/invalid_reasons"
      # UNCOMMENT IF EMAIL SHOULD ONLY BE SENT FOR ERRORS
      # - id: errors_only
      #   default: true
    out: [finished]

  annotate_validation_with_output:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/annotate_submission.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: annotation_values
        source: "#validate/results"
      - id: to_public
        default: true
      - id: force
        default: true
      - id: synapse_config
        source: "#synapseConfig"
    out: [finished]

  check_status:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/check_status.cwl
    in:
      - id: status
        source: "#validate/status"
      - id: previous_annotation_finished
        source: "#annotate_validation_with_output/finished"
      - id: previous_email_finished
        source: "#validation_email/finished"
    out: [finished]
 
  archive:
    run: archive.cwl
    in:
      - id: synapse_config
        source: "#synapseConfig"
      - id: submissionid
        source: "#submissionId"
      - id: admin
        source: "#admin"
      - id: check_validation_finished 
        source: "#check_status/finished"
    out:
      - id: results

  annotate_archive_with_output:
    run: https://raw.githubusercontent.com/Sage-Bionetworks/ChallengeWorkflowTemplates/v3.1/cwl/annotate_submission.cwl
    in:
      - id: submissionid
        source: "#submissionId"
      - id: annotation_values
        source: "#archive/results"
      - id: to_public
        default: true
      - id: force
        default: true
      - id: synapse_config
        source: "#synapseConfig"
      - id: previous_annotation_finished
        source: "#annotate_validation_with_output/finished"
    out: [finished]
