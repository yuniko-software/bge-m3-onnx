#!/usr/bin/env python3
"""
Generate performance test dataset for BGE-M3 embedding benchmarks
Creates 1000 diverse texts with varying lengths, languages, and domains
"""

import json
import os
import random
from typing import List, Dict, Any

# Sample texts for different categories and languages
SAMPLE_TEXTS = {
    "short_english": [
        "High-performance laptop with SSD storage",
        "Best coffee in downtown area",
        "Quick delivery service available",
        "Premium quality headphones on sale",
        "Fast internet connection required",
        "Excellent customer service experience",
        "Modern apartment with city view",
        "Professional web development services",
        "Fresh organic vegetables daily",
        "Reliable cloud storage solution",
        "Advanced machine learning algorithms",
        "Secure payment processing system",
        "Innovative mobile app design",
        "Efficient project management tools",
        "Sustainable energy solutions",
        "Expert financial consulting",
        "Creative digital marketing strategies",
        "Comprehensive health insurance plans",
        "State-of-the-art fitness equipment",
        "Premium educational resources"
    ],
    
    "medium_english": [
        "This innovative smartphone features a cutting-edge camera system with advanced AI capabilities, delivering exceptional photo quality in any lighting condition. The device includes a powerful processor, long-lasting battery, and premium build quality that sets new standards in mobile technology.",
        "Our comprehensive software development service offers end-to-end solutions for businesses of all sizes. We specialize in creating scalable applications using modern frameworks and technologies, ensuring optimal performance, security, and user experience across all platforms.",
        "The latest research in artificial intelligence demonstrates significant breakthroughs in natural language processing and computer vision. These advancements enable more sophisticated applications in healthcare, finance, education, and autonomous systems.",
        "Climate change represents one of the most pressing challenges of our time, requiring immediate action from governments, businesses, and individuals. Sustainable practices, renewable energy adoption, and carbon reduction strategies are essential for environmental preservation.",
        "Modern data analytics platforms provide powerful insights that drive business decision-making. By leveraging machine learning algorithms and statistical models, organizations can identify trends, predict outcomes, and optimize operations for maximum efficiency.",
        "The evolution of remote work has transformed traditional business operations, enabling global collaboration and flexible employment arrangements. Companies are investing in digital infrastructure and communication tools to support distributed teams effectively.",
        "Advanced cybersecurity measures are crucial for protecting sensitive information in today's digital landscape. Multi-factor authentication, encryption protocols, and continuous monitoring systems help safeguard against evolving cyber threats.",
        "E-commerce platforms continue to revolutionize retail by offering personalized shopping experiences, seamless payment processing, and efficient logistics solutions. Consumer behavior analytics drive targeted marketing and inventory management strategies.",
        "Educational technology innovations are reshaping how students learn and teachers instruct. Interactive platforms, virtual reality experiences, and adaptive learning systems enhance engagement and improve educational outcomes.",
        "Healthcare digitization improves patient care through electronic medical records, telemedicine services, and diagnostic AI systems. These technologies enable more accurate diagnoses, efficient treatments, and better health monitoring capabilities."
    ],
    
    "long_english": [
        """The rapid advancement of artificial intelligence and machine learning technologies has fundamentally transformed numerous industries and aspects of daily life. From healthcare diagnostics that can identify diseases with unprecedented accuracy to autonomous vehicles that navigate complex urban environments, AI systems are becoming increasingly sophisticated and ubiquitous. 
        Natural language processing models, such as large language models, have achieved remarkable capabilities in understanding and generating human-like text, enabling applications in customer service, content creation, translation, and educational assistance. Computer vision systems can now recognize objects, faces, and complex scenes with accuracy that often surpasses human performance, leading to innovations in security, manufacturing quality control, and medical imaging analysis.
        However, the integration of AI technologies also raises important ethical considerations and challenges. Issues of bias in algorithmic decision-making, privacy concerns related to data collection and usage, and the potential displacement of human workers require careful consideration and proactive solutions. Organizations must balance the benefits of AI adoption with responsible development practices that ensure fairness, transparency, and accountability.
        The future of AI development depends on continued research in areas such as explainable AI, federated learning, and quantum computing integration. As these technologies mature, they will likely enable even more powerful and efficient AI systems that can tackle complex global challenges including climate change, healthcare accessibility, and educational inequality while maintaining ethical standards and human-centered design principles.""",
        """Cloud computing has revolutionized the way businesses operate by providing scalable, flexible, and cost-effective IT infrastructure solutions. Organizations of all sizes can now access enterprise-grade computing resources, storage capabilities, and software applications without the need for significant upfront capital investments in physical hardware and data centers.
        The three primary cloud service models - Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS) - offer different levels of abstraction and management responsibility. IaaS provides virtualized computing resources including servers, storage, and networking components, allowing businesses to scale their infrastructure dynamically based on demand. PaaS offerings include development platforms, databases, and middleware services that enable developers to build and deploy applications without managing underlying infrastructure complexities.
        SaaS applications have become integral to modern business operations, providing ready-to-use software solutions for customer relationship management, human resources, accounting, and collaboration. These cloud-based applications offer automatic updates, cross-platform accessibility, and collaborative features that enhance productivity and reduce IT maintenance overhead.
        Security considerations in cloud computing require comprehensive strategies that address data protection, access control, compliance requirements, and shared responsibility models between cloud providers and customers. Multi-cloud and hybrid cloud architectures are becoming increasingly popular as organizations seek to avoid vendor lock-in, optimize costs, and maintain control over sensitive data while leveraging the benefits of cloud technologies."""
    ],
    
    "short_spanish": [
        "Servicio de entrega rápida disponible",
        "Restaurante italiano auténtico",
        "Tecnología innovadora para empresas",
        "Curso de programación online",
        "Productos ecológicos de calidad",
        "Consultoría financiera profesional",
        "Diseño web moderno y responsive",
        "Clases de español para extranjeros",
        "Equipos deportivos de alta gama",
        "Soluciones de marketing digital"
    ],
    
    "medium_spanish": [
        "Nuestra plataforma de comercio electrónico ofrece una experiencia de compra personalizada con recomendaciones inteligentes, procesamiento seguro de pagos y entrega rápida. Utilizamos tecnología avanzada para garantizar la satisfacción del cliente y optimizar las operaciones comerciales.",
        "El desarrollo sostenible requiere un enfoque integral que combine crecimiento económico, protección ambiental y equidad social. Las empresas modernas deben adoptar prácticas responsables que minimicen su impacto ecológico mientras generan valor para todas las partes interesadas.",
        "La transformación digital ha cambiado fundamentalmente la forma en que las organizaciones operan y se relacionan con sus clientes. La adopción de nuevas tecnologías permite mejorar la eficiencia, reducir costos y crear nuevas oportunidades de negocio en mercados competitivos."
    ],
    
    "short_chinese": [
        "高性能笔记本电脑优惠销售",
        "专业软件开发服务",
        "在线教育平台课程",
        "绿色环保产品推荐",
        "智能家居解决方案",
        "金融投资咨询服务",
        "现代化办公空间租赁",
        "健康医疗检查套餐",
        "国际物流配送服务",
        "人工智能技术应用"
    ],
    
    "medium_chinese": [
        "我们的人工智能平台采用最新的深度学习技术，为企业提供智能化解决方案。通过大数据分析和机器学习算法，帮助客户优化业务流程，提高运营效率，实现数字化转型目标。",
        "电子商务的快速发展改变了传统零售模式，消费者可以通过移动设备随时随地购买商品。我们致力于构建安全可靠的在线购物平台，提供优质的客户服务和便捷的支付方式。"
    ],
    
    "short_russian": [
        "Быстрая доставка по всему городу",
        "Профессиональные IT-услуги",
        "Качественное медицинское обслуживание",
        "Современные образовательные программы",
        "Экологически чистые продукты",
        "Надежные финансовые решения",
        "Инновационные технологии для бизнеса",
        "Эффективные маркетинговые стратегии"
    ],
    
    "medium_russian": [
        "Цифровая трансформация предприятий требует комплексного подхода к внедрению новых технологий. Наша команда экспертов поможет оптимизировать бизнес-процессы, повысить эффективность работы и обеспечить конкурентные преимущества на рынке.",
        "Развитие искусственного интеллекта открывает новые возможности для автоматизации и оптимизации различных сфер деятельности. Машинное обучение и аналитика больших данных становятся ключевыми инструментами современного бизнеса."
    ],
    
    "short_arabic": [
        "خدمات التوصيل السريع متوفرة",
        "تطوير البرمجيات المتقدمة",
        "منتجات عضوية عالية الجودة",
        "استشارات مالية احترافية",
        "حلول التكنولوجيا المبتكرة",
        "خدمات التعليم الإلكتروني",
        "معدات رياضية متطورة"
    ],
    
    "medium_arabic": [
        "تقدم منصتنا الرقمية حلولاً متكاملة للشركات الراغبة في التحول الرقمي. نستخدم أحدث التقنيات في الذكاء الاصطناعي وتحليل البيانات لتحسين العمليات التجارية وزيادة الكفاءة التشغيلية.",
        "التجارة الإلكترونية تشهد نمواً متسارعاً في المنطقة، مما يتطلب استثمارات في التقنيات الحديثة وتطوير المنصات الرقمية. نحن نساعد الشركات على بناء حضور قوي عبر الإنترنت وتحقيق أهدافها التجارية."
    ]
}

# Technical and domain-specific content
TECHNICAL_TEXTS = {
    "short": [
        "Machine learning model optimization techniques",
        "Cloud-native microservices architecture",
        "Blockchain consensus algorithm implementation",
        "Neural network backpropagation analysis",
        "Distributed system fault tolerance",
        "API rate limiting and throttling",
        "Database indexing performance tuning",
        "Container orchestration with Kubernetes",
        "Real-time data streaming pipelines",
        "Cybersecurity threat detection algorithms"
    ],
    
    "medium": [
        "The implementation of transformer architectures has revolutionized natural language processing tasks. Self-attention mechanisms enable models to capture long-range dependencies in sequential data, leading to improved performance in machine translation, text summarization, and language understanding applications.",
        "Microservices architecture patterns facilitate scalable application development by decomposing monolithic systems into smaller, independently deployable services. Service mesh technologies like Istio provide traffic management, security, and observability features for complex distributed systems.",
        "Edge computing brings computational resources closer to data sources, reducing latency and bandwidth requirements. This paradigm is particularly beneficial for IoT applications, autonomous vehicles, and real-time analytics where immediate processing is critical.",
        "DevOps practices integrate development and operations teams to accelerate software delivery cycles. Continuous integration and continuous deployment (CI/CD) pipelines automate testing, building, and deployment processes, improving code quality and reducing time-to-market."
    ]
}

def generate_variations(base_text: str, num_variations: int = 3) -> List[str]:
    """Generate slight variations of a base text"""
    variations = [base_text]
    
    # Simple variations by adding prefixes/suffixes
    prefixes = ["", "New: ", "Featured: ", "Popular: ", "Recommended: "]
    suffixes = ["", ".", " - Limited time offer!", " Available now.", " Contact us for details."]
    
    for i in range(num_variations - 1):
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        variation = f"{prefix}{base_text}{suffix}".strip()
        if variation not in variations:
            variations.append(variation)
    
    return variations

def categorize_text_length(text: str) -> str:
    """Categorize text by length (word count)"""
    word_count = len(text.split())
    if word_count <= 15:
        return "short"
    elif word_count <= 100:
        return "medium"
    else:
        return "long"

def detect_language(category: str) -> str:
    """Extract language from category name"""
    if "english" in category:
        return "english"
    elif "spanish" in category:
        return "spanish"
    elif "chinese" in category:
        return "chinese"
    elif "russian" in category:
        return "russian"
    elif "arabic" in category:
        return "arabic"
    else:
        return "english"

def detect_domain(text: str) -> str:
    """Simple domain detection based on keywords"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["ai", "machine learning", "algorithm", "neural", "model", "data"]):
        return "technology"
    elif any(word in text_lower for word in ["business", "service", "customer", "company", "solution"]):
        return "business"
    elif any(word in text_lower for word in ["health", "medical", "doctor", "treatment", "healthcare"]):
        return "healthcare"
    elif any(word in text_lower for word in ["education", "learning", "course", "student", "teaching"]):
        return "education"
    elif any(word in text_lower for word in ["cloud", "software", "development", "programming", "api"]):
        return "technology"
    else:
        return "general"

def generate_test_dataset(target_size: int = 1000) -> List[Dict[str, Any]]:
    """Generate the complete test dataset"""
    dataset = []
    
    # Generate texts from predefined samples
    for category, texts in SAMPLE_TEXTS.items():
        for text in texts:
            # Generate a few variations of each text
            variations = generate_variations(text, 2)
            for variation in variations:
                if len(dataset) >= target_size:
                    break
                
                dataset.append({
                    "text": variation,
                    "length_category": categorize_text_length(variation),
                    "language": detect_language(category),
                    "domain": detect_domain(variation),
                    "word_count": len(variation.split()),
                    "char_count": len(variation),
                    "source": "predefined"
                })
        
        if len(dataset) >= target_size:
            break
    
    # Add technical texts
    for length_cat, texts in TECHNICAL_TEXTS.items():
        for text in texts:
            if len(dataset) >= target_size:
                break
            
            variations = generate_variations(text, 2)
            for variation in variations:
                if len(dataset) >= target_size:
                    break
                
                dataset.append({
                    "text": variation,
                    "length_category": categorize_text_length(variation),
                    "language": "english",
                    "domain": "technology",
                    "word_count": len(variation.split()),
                    "char_count": len(variation),
                    "source": "technical"
                })
    
    # Generate additional synthetic texts if needed
    while len(dataset) < target_size:
        # Create synthetic combinations
        base_texts = [item["text"] for item in dataset[:50]]  # Use first 50 as templates
        
        for base_text in base_texts:
            if len(dataset) >= target_size:
                break
            
            # Create a synthetic variation
            words = base_text.split()
            if len(words) > 5:
                # Take random subset and add context
                subset_size = random.randint(3, min(8, len(words)))
                subset = random.sample(words, subset_size)
                synthetic_text = " ".join(subset) + " with advanced features"
                
                dataset.append({
                    "text": synthetic_text,
                    "length_category": categorize_text_length(synthetic_text),
                    "language": "english",
                    "domain": detect_domain(synthetic_text),
                    "word_count": len(synthetic_text.split()),
                    "char_count": len(synthetic_text),
                    "source": "synthetic"
                })
    
    # Trim to exact target size and shuffle
    dataset = dataset[:target_size]
    random.shuffle(dataset)
    
    # Add index to each item
    for i, item in enumerate(dataset):
        item["id"] = i
    
    return dataset

def save_dataset(dataset: List[Dict[str, Any]], output_dir: str):
    """Save the dataset and metadata"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main dataset
    dataset_path = os.path.join(output_dir, "test_texts.json")
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    # Generate and save metadata
    metadata = {
        "total_texts": len(dataset),
        "statistics": {
            "length_distribution": {},
            "language_distribution": {},
            "domain_distribution": {},
            "source_distribution": {},
            "word_count_stats": {},
            "char_count_stats": {}
        }
    }
    
    # Calculate distributions
    for field in ["length_category", "language", "domain", "source"]:
        values = [item[field] for item in dataset]
        distribution = {}
        for value in set(values):
            distribution[value] = values.count(value)
        metadata["statistics"][f"{field[:-9] if field.endswith('_category') else field}_distribution"] = distribution
    
    # Calculate word and character count statistics
    word_counts = [item["word_count"] for item in dataset]
    char_counts = [item["char_count"] for item in dataset]
    
    metadata["statistics"]["word_count_stats"] = {
        "min": min(word_counts),
        "max": max(word_counts),
        "avg": sum(word_counts) / len(word_counts),
        "median": sorted(word_counts)[len(word_counts) // 2]
    }
    
    metadata["statistics"]["char_count_stats"] = {
        "min": min(char_counts),
        "max": max(char_counts),
        "avg": sum(char_counts) / len(char_counts),
        "median": sorted(char_counts)[len(char_counts) // 2]
    }
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"Dataset saved to {dataset_path}")
    print(f"Metadata saved to {metadata_path}")
    
    # Print summary
    print(f"\nDataset Summary:")
    print(f"Total texts: {metadata['total_texts']}")
    print(f"Length distribution: {metadata['statistics']['length_distribution']}")
    print(f"Language distribution: {metadata['statistics']['language_distribution']}")
    print(f"Domain distribution: {metadata['statistics']['domain_distribution']}")
    print(f"Average word count: {metadata['statistics']['word_count_stats']['avg']:.1f}")

def main():
    """Main function to generate the test dataset"""
    # Set random seed for reproducibility
    random.seed(42)
    
    # Find repository root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(script_dir))
    
    # Create output directory
    output_dir = os.path.join(repo_root, "samples", "performance_data")
    
    print("Generating BGE-M3 performance test dataset...")
    print(f"Output directory: {output_dir}")
    
    # Generate dataset
    dataset = generate_test_dataset(1000)
    
    # Save dataset and metadata
    save_dataset(dataset, output_dir)
    
    print("\nDataset generation completed successfully!")

if __name__ == "__main__":
    main()
