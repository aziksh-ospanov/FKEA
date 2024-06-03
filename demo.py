from FKEA.metric.FKEA import FKEA_Evaluator
from FKEA.datasets.ImageFilesDataset import ImageFilesDataset
from FKEA.datasets.TextFilesDataset import TextFilesDataset
import pickle
#%%
if __name__ == '__main__':
    num_samples = 100
    sigma = 20
    rff_dim = 4000
    
    # Image Clustering
    result_name = 'your_result_name'
    fe = 'dinov2' # feature embedding for img
    
    FKEA = FKEA_Evaluator(logger_path='./logs', batchsize=8, sigma=sigma, num_samples=num_samples, result_name=result_name, rff_dim=rff_dim)
    FKEA.set_feature_extractor(fe, save_path='./save')

    img_pth = r'\\wsl.localhost\Ubuntu\home\azimospanov\development\datasets\imagenet_100k\train_100000\train'
    novel_dataset = ImageFilesDataset(img_pth, name=result_name, extension='JPEG')

    FKEA.rff_clustering_modes_of_dataset(novel_dataset)
    alpha = [1, 1.0, 2, 2.01, 'inf']
    print(FKEA.f_diversity_scores(novel_dataset, alpha=alpha))
    print(FKEA.non_f_diversity_scores(novel_dataset, alpha=alpha))
    
    
    # # Text Clustering
    # result_name = 'text'
    # fe = 'gpt-large'
    
    # FKEA = FKEA_Evaluator(logger_path='./logs', batchsize=16, sigma=sigma, num_samples=num_samples, result_name=result_name, rff_dim=rff_dim, 
    #                       api_key='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')
    # FKEA.set_feature_extractor(fe, save_path='./save')
    
    # txt_path = r"C:\Users\admin\development\text-generation\essay_evaluation\synthetic_countries_by_attractions\copy\txt_files"
    # # with open(txt_path, 'r', encoding='utf-8') as f:
    # #     texts = f.readlines()
    
    # # embedding_pth = 'embedding/path/*.pickle' # In this example, we store embeddings as pickle file, should be convertible to torch tensor
    # # with open(embedding_pth, 'rb') as f:
    # #     embeddings = pickle.load(f)
    # # assert len(texts) == embeddings.shape[0]
    # # assert embeddings.shape[0] >= num_samples
    
    # novel_dataset = TextFilesDataset(txt_path).dataset
    
    # FKEA.rff_text_clustering_modes_of_dataset(novel_dataset)
    # print(FKEA.f_diversity_scores(novel_dataset))
    # # print(FKEA.non_f_diversity_scores(novel_dataset))
    
    # # FKEA.rff_text_clustering_modes_of_dataset(texts, test_feats = embeddings)
    # # print(FKEA.f_diversity_scores(texts, test_feats=embeddings))
    # # print(FKEA.non_f_diversity_scores(texts, test_feats=embeddings))


