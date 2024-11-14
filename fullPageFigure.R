# Load necessary libraries
library(ggplot2)
library(cowplot)

# Define plot settings
person_id <- 'P7_453'
plot_folder <- "/Users/jg/Desktop/upper_limb/paper_figures-2"
data_folder <- "/Users/jg/Desktop/upper_limb/paper_data-2"
save_dir <- file.path(plot_folder, 'fullPage')
dir.create(save_dir, recursive = TRUE, showWarnings = FALSE)

plot_settings <- list(
  list(indices = c(4, 5), plot_recordings = c('keyOpCl', 'thumbFlEx', 'thumbAbAd', 'pinchOpCl')),
  list(indices = c(2, 6), plot_recordings = c('wristFlHandCl', 'wristFlEx', 'handOpCl', 'fingersFlEx')),
  list(indices = c(0, 1), plot_recordings = c('indexFlDigitsEx', 'indexFlEx', 'mrpFlEx', 'pointOpCl'))
)

colors <- c('#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', '#00FFFF', '#FF00FF')
plots <- list()  # Initialize empty list for unique plots

# Loop to create and store each plot individually
setting_id <- 0
for (setting in plot_settings) {
  indices <- setting$indices
  group_recordings <- setting$plot_recordings
  recording_id <- 0
  for (recording in group_recordings) {
    gt_path <- file.path(data_folder, 'trajectories', person_id, recording, 'GT', 'pred.csv')
    after_init_path <- file.path(data_folder, 'trajectories', person_id, recording, sprintf("perturb_%s", FALSE), 'before_online', 'pred.csv')
    before_path <- file.path(data_folder, 'trajectories', person_id, recording, sprintf("perturb_%s", TRUE), 'before_online', 'pred.csv')
    after_path <- file.path(data_folder, 'trajectories', person_id, recording, sprintf("perturb_%s", TRUE), 'after_online', 'pred.csv')
    
    gt <- as.data.frame(read.csv(gt_path))
    after_init <- as.data.frame(read.csv(after_init_path))
    before <- as.data.frame(read.csv(before_path))
    after <- as.data.frame(read.csv(after_path))
    
    x_values <- 1:nrow(gt)
    
    # After Initial Training ##############################################
    plot_after_init <- ggplot() +
      geom_line(data = gt, aes(x = X, y = eval(sym(paste0('X', indices[1])))), color = colors[indices[1] + 1]) +
      geom_line(data = gt, aes(x = X, y = eval(sym(paste0('X', indices[2])))), color = colors[indices[2] + 1]) +
      geom_line(data = after_init, aes(x = X, y = eval(sym(paste0('X', indices[1])))), color = colors[indices[1] + 1], linetype = 'dashed') +
      geom_line(data = after_init, aes(x = X, y = eval(sym(paste0('X', indices[2])))), color = colors[indices[2] + 1], linetype = 'dashed') +
      ylab('Normalized Angle') +
      theme_bw() 
    if (setting_id == 0) {
      plot_after_init <- plot_after_init + ggtitle("After Initial Training")
    }
    if (recording_id == 3) {
      plot_after_init <- plot_after_init + xlab('Time [seconds]')
    }
    else {
      plot_after_init <- plot_after_init +
        theme(
          legend.position = "none",
          axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank()
        )
    }
    plots <- c(plots, list(plot_after_init))  # Add to plots list
    
    # After Perturbation ##############################################
    plot_after_perturb <- ggplot() +
      geom_line(data = gt, aes(x = X, y = eval(sym(paste0('X', indices[1])))), color = colors[indices[1] + 1]) +
      geom_line(data = gt, aes(x = X, y = eval(sym(paste0('X', indices[2])))), color = colors[indices[2] + 1]) +
      geom_line(data = before, aes(x = X, y = eval(sym(paste0('X', indices[1])))), color = colors[indices[1] + 1], linetype = 'dashed') +
      geom_line(data = before, aes(x = X, y = eval(sym(paste0('X', indices[2])))), color = colors[indices[2] + 1], linetype = 'dashed') +
      ylab('Normalized Angle') +
      theme_bw() +
      theme(
        legend.position = "none",
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
    
    if (setting_id == 0) {
      plot_after_perturb <- plot_after_perturb + ggtitle("After Perturbation")
    }
    if (recording_id == 3) {
      plot_after_perturb <- plot_after_perturb + xlab('Time [seconds]')
    }
    else {
      plot_after_perturb <- plot_after_perturb +
        theme(
          axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank()
        )
    }
    plots <- c(plots, list(plot_after_perturb))  # Add to plots list
    
    # After Online Training ##############################################
    plot_after_online <- ggplot() +
      geom_line(data = gt, aes(x = X, y = eval(sym(paste0('X', indices[1])))), color = colors[indices[1] + 1]) +
      geom_line(data = gt, aes(x = X, y = eval(sym(paste0('X', indices[2])))), color = colors[indices[2] + 1]) +
      geom_line(data = after, aes(x = X, y = eval(sym(paste0('X', indices[1])))), color = colors[indices[1] + 1], linetype = 'dashed') +
      geom_line(data = after, aes(x = X, y = eval(sym(paste0('X', indices[2])))), color = colors[indices[2] + 1], linetype = 'dashed') +
      ylab('Normalized Angle') +
      theme_bw() +
      theme(
        legend.position = "none",
        axis.title.y = element_blank(),
        axis.text.y = element_blank(),
        axis.ticks.y = element_blank())
    
    if (setting_id == 0) {
      plot_after_online <- plot_after_online + ggtitle("After Online Training")
    }
    if (recording_id == 3) {
      plot_after_online <- plot_after_online + labs(x = 'Time [seconds]')
    }
    else {
      plot_after_online <- plot_after_online +
        theme(
          axis.title.x = element_blank(),
          axis.text.x = element_blank(),
          axis.ticks.x = element_blank()
        )
    }
    plots <- c(plots, list(plot_after_online))  # Add to plots list
    
    setting_id <- setting_id + 1
    recording_id <- recording_id + 1
  }
}

# Arrange and save the grid of unique plots using cowplot::plot_grid
num_rows <- ceiling(length(plots) / 3)
grid <- plot_grid(plotlist = plots, ncol = 3, align = "v")

# Save the full grid as one image
ggsave(
  plot = grid,
  filename = file.path(save_dir, 'all_recordings.png'),
  width = 18,
  height = 5 * num_rows,
  dpi = 300,
  limitsize = FALSE
)
