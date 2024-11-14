# Load necessary libraries
library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)

# Define directories and settings
person_id <- "P7_453"
plot_folder <- "/Users/jg/Desktop/upper_limb/paper_figures-2"
data_folder <- "/Users/jg/Desktop/upper_limb/paper_data-2"
save_dir <- file.path(plot_folder, "fullPage")
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

# Plot settings
plot_settings <- list(
  list(indices = c(4, 5), plot_recordings = c("keyOpCl", "thumbFlEx", "thumbAbAd", "pinchOpCl")),
  list(indices = c(2, 6), plot_recordings = c("wristFlHandCl", "wristFlEx", "handOpCl", "fingersFlEx")),
  list(indices = c(0, 1), plot_recordings = c("indexFlDigitsEx", "indexFlEx", "mrpFlEx", "pointOpCl")) 
)
# colors <- c('#D55E00', '#0072B2', '#009E73', '#E69F00', '#F9627D', '#56B4E9', '#CC79A7')
# colors <- c("#a0e85b", "#154975", "#70c6ca", "#3337a6", "#bbc3fe", "#0b6d33")
colors <- c("#53baa0", "#f89231", "#63a122", "#8323dc", "#3ba7e5", "#d77cdd", "#35618f")
color_mapping <- c("#53baa0" = "indexAng          ", "#f89231" = "midAng", "#63a122" = "ringAng            ", "#8323dc" = "pinkyAng", "#3ba7e5" = "thumbInPlane ", "#d77cdd" = "thumbOutPlane", "#35618f" = "wristFlex")


# Initialize empty data frame for plotting
plot_data <- data.frame()
setting_id <- 0

# Loop through settings and recordings to load data
for (setting in plot_settings) {
  indices <- setting$indices
  group_recordings <- setting$plot_recordings
  recording_id <- 0

  for (recording in group_recordings) {
    file_paths <- list(
      gt = file.path(data_folder, "trajectories", person_id, recording, "GT", "pred.csv"),
      after_init = file.path(data_folder, "trajectories", person_id, recording, sprintf("perturb_%s", FALSE), "before_online", "pred.csv"),
      before = file.path(data_folder, "trajectories", person_id, recording, sprintf("perturb_%s", TRUE), "before_online", "pred.csv"),
      after = file.path(data_folder, "trajectories", person_id, recording, sprintf("perturb_%s", TRUE), "after_online", "pred.csv")
    )

    data_list <- lapply(file_paths, read.csv, header = FALSE)
    X <- 1:nrow(data_list$gt)

    # Helper function to prepare long format data
    prep_long_data <- function(data, type, line_type) {
      data %>%
        mutate(Time = X) %>%
        select(Time, V1 = !!sym(paste0("V", indices[1] + 1)), V2 = !!sym(paste0("V", indices[2] + 1))) %>%
        pivot_longer(cols = c(V1, V2), names_to = "Variable", values_to = "Value") %>%
        mutate(
          Data_Type = type,
          Line_Type = line_type,
          Setting_ID = setting_id,
          Recording_ID = recording_id,
          Variable_Index = ifelse(Variable == "V1", indices[1], indices[2]),
          Color = colors[Variable_Index + 1],
          Recording_Name = recording
        )
    }

    # Prepare and combine data phases
    data_after_init <- bind_rows(prep_long_data(data_list$gt, "gt", "solid"), prep_long_data(data_list$after_init, "after_init", "dashed")) %>%
      mutate(Phase = "After Initial Training")
    data_before <- bind_rows(prep_long_data(data_list$gt, "gt", "solid"), prep_long_data(data_list$before, "before", "dashed")) %>%
      mutate(Phase = "After Perturbation")
    data_after <- bind_rows(prep_long_data(data_list$gt, "gt", "solid"), prep_long_data(data_list$after, "after", "dashed")) %>%
      mutate(Phase = "After Online Training")

    # Combine data into plot_data
    plot_data <- bind_rows(plot_data, data_after_init, data_before, data_after)
    recording_id <- recording_id + 1
  }
  setting_id <- setting_id + 1
}

# Set factor levels for correct plotting order
plot_data <- plot_data %>%
  mutate(
    Phase = factor(Phase, levels = c("After Initial Training", "After Perturbation", "After Online Training")), # nolint: line_length_linter.
    Recording_Name = factor(Recording_Name, levels = unique(Recording_Name))
  )

# Define data subsets for plotting
data_first_set <- plot_data %>% filter(Recording_Name %in% levels(Recording_Name)[1:4])
data_second_set <- plot_data %>% filter(Recording_Name %in% levels(Recording_Name)[5:8])
data_last_set <- plot_data %>% filter(!Recording_Name %in% levels(Recording_Name)[1:8])



  
  
create_plot <- function(data) {
  ggplot(data, aes(x = Time, y = Value, color = Color, linetype = Line_Type)) +
    geom_line(size = 1, alpha = 0.8) + # Slightly thicker lines with transparency
    scale_color_manual(
      values = unique(data$Color), # Uses hex color codes directly from the dataset
      labels = color_mapping,
      guide = guide_legend(
        # override.aes = list(shape = 16, size = 4), 
        title = element_blank()
        ) # Change color legend to circles
    ) +
    scale_linetype_manual(
      values = c("solid" = "solid", "dashed" = "11"), # Define linetype mappings
      labels = c("solid" = "Target", "dashed" = "Prediction"), # Custom labels for linetypes
      guide = guide_legend(
        # override.aes = list(shape = NA, size = 4), 
        title = element_blank()
      ) # Override to keep linetype as lines
    ) +    
    facet_grid(
      rows = vars(Recording_Name), 
      cols = vars(Phase), 
      switch = "y"
    ) +
    scale_y_continuous(
      limits = c(-1.3, 1.3), 
      position = "right",
      breaks = c(-1, 0, 1)
    ) +
    scale_x_continuous(
      breaks = seq(0, 360, by = 120),  # Specify breaks every 60 units
      labels = c(0, 2, 4, 6)           # Label these breaks as 0 to 6
    ) +    
    theme(
      legend.position = c(0.9834, 0.7892), # Adjust to move it right and up slightly
      legend.justification = c("right", "top"), # Aligns legend top-right corner to position

      legend.direction = "vertical", # Horizontal legend
      legend.box = "horizontal", # Display color and linetype legends in one line
      legend.margin = margin(t = -10, r = 2, b = 0, l = 0),
      legend.key = element_blank(), # Removes background behind legend markers
      legend.spacing = unit(0, "lines"), # Remove spacing between legend items
      legend.background = element_rect(fill="gray90"), # Adds border around legend

      plot.margin = margin(t = 0, r = 0, b = 28, l = 0), # Adjust plot margin to fit the legend
      strip.text.y.left = element_text(size = 10),
      strip.placement = "outside",
      strip.text = element_text(size = 12, face = "bold"),
      # strip.background = element_blank(), # Remove strip background

      panel.grid.minor = element_blank(), # Removes minor grid lines
      panel.grid.major = element_line(size = 0.3, color = "gray90"), # Lightens major grid lines
      panel.background = element_rect(fill = "gray98", color = NA), # Light gray background within the panel
      # panel.background = element_blank(), # Remove strip background

      axis.ticks.length = unit(0, "pt"), # Removes ticks for smooth appearance
      # panel.spacing = unit(0, "lines"), # Remove spacing between panels


    ) +
    labs(x = "Time (s)", y = "Normalized Angle")
}


# Create individual plots
p1 <- create_plot(data_first_set)
p2 <- create_plot(data_second_set)
p3 <- create_plot(data_last_set)

# Arrange and save combined plot
combined_plot <- grid.arrange(p1, p2, p3, ncol = 1)
ggsave(
  plot = combined_plot,
  filename = file.path(save_dir, "trajectories.png"),
  width = 18,
  height = 18 * 1.2,
  dpi = 600,
  limitsize = FALSE
)
